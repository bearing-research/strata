"""BigQuery driver adapter, backed by ``adbc-driver-bigquery``.

Freshness reads ``last_modified_time`` from each dataset's legacy ``__TABLES__``
view (``INFORMATION_SCHEMA.TABLES`` has no such column); the schema fingerprint
reads ``INFORMATION_SCHEMA.COLUMNS``.

Read-only is enforced by credentials, since BigQuery has no session read-only
flag: read cells should use a ``credentials_path`` service account with
``roles/bigquery.dataViewer`` + ``roles/bigquery.jobUser``, write cells a
``write_credentials_path`` one with ``roles/bigquery.dataEditor`` (without it,
writes use the read credentials).

Streaming inserts make ``last_modified_time`` lag by up to about 90 minutes, so
freshness is underestimated; such cells should use ``# @cache session``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

from strata.notebook.sql.adapter import (
    AdapterCapabilities,
    ColumnInfo,
    FreshnessToken,
    QualifiedTable,
    SchemaFingerprint,
    TableSchema,
    hash_connection_identity,
)
from strata.notebook.sql.registry import register_adapter
from strata.notebook.sql.time_travel import iso_utc, pin_tables, plus

# The shortest window a dataset can be configured with.
_GUARANTEED_TIME_TRAVEL = timedelta(hours=48)

_CAPABILITIES = AdapterCapabilities(
    per_table_freshness=True,
    # No per-table snapshot id, but ``FOR SYSTEM_TIME AS OF`` reads a table
    # as of a timestamp, so a snapshot is a timestamp (``time_travel.py``).
    supports_snapshot=True,
    # ``__TABLES__`` and ``INFORMATION_SCHEMA`` aren't frozen inside a transaction.
    needs_separate_probe_conn=False,
)

# Project and dataset IDs follow different GCP rules, so each is validated
# on its own. The regexes defend against splice injection; they don't
# enforce GCP's length limits.
_PROJECT_ID_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_DATASET_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_valid_project_id(value: str) -> bool:
    return bool(_PROJECT_ID_RE.match(value))


def _is_valid_dataset_id(value: str) -> bool:
    return bool(_DATASET_ID_RE.match(value))


def _spec_attr(spec: Any, key: str) -> Any:
    """Read a top-level field off a ``ConnectionSpec`` safely.

    ``BaseModel.schema`` shadows extras, so prefer ``model_extra`` and reject
    bound-method shadows (as in the Snowflake adapter).
    """
    extras = getattr(spec, "model_extra", None) or {}
    if key in extras:
        return extras.get(key)
    value = getattr(spec, key, None)
    if callable(value) and getattr(value, "__self__", None) is not None:
        return None
    return value


def _credentials_principal(path_value: Any) -> str | None:
    """Read ``client_email`` from a service-account JSON file, or None if unreadable.

    BigQuery keys visibility on the principal, so it goes into the cache
    identity; when it cannot be read, the path is folded instead.
    """
    if not isinstance(path_value, str) or not path_value:
        return None
    try:
        data = json.loads(Path(path_value).read_text())
    except (OSError, ValueError):
        return None
    if isinstance(data, dict):
        email = data.get("client_email")
        if isinstance(email, str) and email:
            return email
    return None


class BigQueryAdapter:
    """ADBC-backed driver adapter for Google BigQuery."""

    name = "bigquery"
    sqlglot_dialect = "bigquery"
    capabilities = _CAPABILITIES

    def __init__(
        self,
        *,
        connect_fn: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        # Test seam: a fake taking the kwargs production would hand to ADBC.
        self._connect_fn = connect_fn

    # --- identity ---------------------------------------------------------

    def canonicalize_connection_id(self, spec: Any, *, read_only: bool = True) -> str:
        """Hash identity-shaping fields, excluding secrets.

        Folds project_id, dataset_id and the service account principal: read
        cells use ``credentials_path``'s, write cells ``write_credentials_path``'s
        (else ``credentials_path``'s), so swapping the write SA leaves read caches
        alone. A path stands in when the principal cannot be read.

        With no credentials path the adapter uses ambient ADC, whose principal is
        unknown without a network call, so an ``ambient_adc`` sentinel is folded;
        the cache can then be shared across machines with different ambient
        principals. Set ``credentials_path`` for a stable identity.
        """
        return hash_connection_identity(
            self.name, self._extract_identity(spec, read_only=read_only)
        )

    def _extract_identity(self, spec: Any, *, read_only: bool = True) -> dict[str, Any]:
        identity: dict[str, Any] = {}

        for key in ("project_id", "dataset_id"):
            value = _spec_attr(spec, key)
            if value is not None:
                identity[key] = str(value)

        # Write-side credentials join only when ``read_only=False``, so changing
        # ``write_credentials_path`` doesn't invalidate read-cell caches.
        ro_path = _spec_attr(spec, "credentials_path")
        if ro_path:
            principal = _credentials_principal(ro_path)
            identity["credentials_principal"] = principal or str(ro_path)

        if not read_only:
            rw_path = _spec_attr(spec, "write_credentials_path")
            if rw_path:
                principal = _credentials_principal(rw_path)
                identity["write_credentials_principal"] = principal or str(rw_path)

        # No credentials: the driver uses ambient ADC, whose principal is unknown
        # without a network call, so flag it. Set ``credentials_path`` for
        # machine-portable cache identity.
        active_creds = (
            ro_path if read_only else (_spec_attr(spec, "write_credentials_path") or ro_path)
        )
        if not active_creds:
            identity["ambient_adc"] = True

        return identity

    # --- connection lifecycle --------------------------------------------

    def open(self, spec: Any, *, read_only: bool) -> Any:
        """Open an ADBC BigQuery connection.

        ``open(read_only=False)`` uses ``write_credentials_path`` when set, else
        ``credentials_path``; the account's IAM grants decide whether DML runs.
        The driver takes keyword arguments, not a URI, and the connect_fn test
        seam mirrors that with a dict.
        """
        ro_creds = _spec_attr(spec, "credentials_path")
        rw_creds = _spec_attr(spec, "write_credentials_path") or ro_creds
        chosen_creds = ro_creds if read_only else rw_creds

        kwargs: dict[str, Any] = {}
        project = _spec_attr(spec, "project_id")
        if project:
            kwargs["adbc.bigquery.sql.project_id"] = str(project)
        dataset = _spec_attr(spec, "dataset_id")
        if dataset:
            kwargs["adbc.bigquery.sql.dataset_id"] = str(dataset)
        if chosen_creds:
            kwargs["adbc.bigquery.sql.auth_type"] = (
                "adbc.bigquery.sql.auth_type.json_credential_file"
            )
            kwargs["adbc.bigquery.sql.auth_credentials"] = str(chosen_creds)

        return self._invoke_connect(kwargs)

    def _invoke_connect(self, kwargs: dict[str, Any]) -> Any:
        if self._connect_fn is not None:
            return self._connect_fn(kwargs)
        try:
            from adbc_driver_bigquery import dbapi as adbc_bigquery
        except ImportError as exc:
            raise RuntimeError(
                "adbc-driver-bigquery is not installed; install with "
                "`uv pip install 'strata-notebook[sql-bigquery]'`"
            ) from exc
        return adbc_bigquery.connect(db_kwargs=kwargs)

    # --- helpers ---------------------------------------------------------

    def _resolve_session_defaults(self, cursor: Any) -> tuple[str | None, str | None]:
        """Read the session's default ``(project, dataset)``, either possibly None.

        From ``@@project_id`` and ``@@dataset_id``; used to resolve unqualified
        tables in probes.
        """
        try:
            cursor.execute("SELECT @@project_id, @@dataset_id")
            row = cursor.fetchone()
            if not row:
                return None, None
            proj = str(row[0]) if row[0] else None
            ds = str(row[1]) if len(row) > 1 and row[1] else None
            return proj, ds
        except Exception:  # noqa: BLE001
            # Some contexts (older ADBC, BigQuery Omni) lack @@dataset_id.
            return None, None

    # --- probes ----------------------------------------------------------

    def snapshot_timestamp(self, conn: Any) -> str:
        with conn.cursor() as cursor:
            cursor.execute("SELECT CURRENT_TIMESTAMP()")
            row = cursor.fetchone()
        return iso_utc(row[0])

    def retention_until(self, conn: Any, tables: list[QualifiedTable], at: str) -> str | None:
        """``at`` plus BigQuery's guaranteed time-travel window.

        A dataset's window is configurable (two to seven days) and lives in a
        region-scoped view this adapter cannot query without a region, so this
        uses the minimum every dataset guarantees.
        """
        return plus(at, _GUARANTEED_TIME_TRAVEL)

    def pin_query(self, sql: str, at: str) -> str:
        return pin_tables(
            sql,
            "bigquery",
            f"FOR SYSTEM_TIME AS OF TIMESTAMP '{at}'",
            "version",
            after_alias=True,
        )

    def probe_freshness(
        self,
        probe_conn: Any,
        tables: list[QualifiedTable],
    ) -> FreshnessToken:
        """Per-table freshness via ``__TABLES__.last_modified_time`` (unix millis).

        One ``__TABLES__`` query per touched (project, dataset). Lags for
        tables receiving streaming inserts (see the module docstring).
        """
        if not tables:
            return FreshnessToken(value=b"")

        by_dataset: dict[tuple[str | None, str | None], list[QualifiedTable]] = {}
        for t in tables:
            by_dataset.setdefault((t.catalog, t.schema), []).append(t)

        h = hashlib.sha256()
        with probe_conn.cursor() as cursor:
            current_proj, current_ds = self._resolve_session_defaults(cursor)

            for (catalog, schema), group in sorted(
                by_dataset.items(),
                key=lambda kv: (kv[0][0] or "") + "/" + (kv[0][1] or ""),
            ):
                effective_project = catalog or current_proj
                effective_dataset = schema or current_ds
                if not effective_project or not effective_dataset:
                    for table in sorted(group, key=lambda t: t.render()):
                        h.update(b"no-dataset:")
                        h.update(table.render().encode())
                        h.update(b"\x00")
                    continue

                if not _is_valid_project_id(effective_project) or not _is_valid_dataset_id(
                    effective_dataset
                ):
                    raise RuntimeError(
                        f"BigQuery identifier {effective_project}.{effective_dataset!r} "
                        "is not valid"
                    )

                # ADBC BigQuery uses ``@name`` named-bind syntax.
                query = (
                    f"SELECT table_id, last_modified_time "
                    f"FROM `{effective_project}.{effective_dataset}.__TABLES__` "
                    f"WHERE table_id = @table_id"
                )
                for table in sorted(group, key=lambda t: t.render()):
                    cursor.execute(query, parameters={"table_id": table.name})
                    row = cursor.fetchone()
                    h.update(effective_project.encode())
                    h.update(b".")
                    h.update(effective_dataset.encode())
                    h.update(b".")
                    h.update(table.name.encode())
                    h.update(b":")
                    if row is None:
                        h.update(b"missing")
                    else:
                        h.update(str(row[0]).encode())
                        h.update(b":")
                        h.update(str(row[1]).encode())
                    h.update(b"\x00")

        return FreshnessToken(value=h.digest())

    def probe_schema(
        self,
        probe_conn: Any,
        tables: list[QualifiedTable],
    ) -> SchemaFingerprint:
        """Per-table schema fingerprint via ``INFORMATION_SCHEMA.COLUMNS``.

        Redundant with ``last_modified_time`` (BigQuery DDL bumps it), kept as a backstop.
        """
        if not tables:
            return SchemaFingerprint(value=b"")

        by_dataset: dict[tuple[str | None, str | None], list[QualifiedTable]] = {}
        for t in tables:
            by_dataset.setdefault((t.catalog, t.schema), []).append(t)

        h = hashlib.sha256()
        with probe_conn.cursor() as cursor:
            current_proj, current_ds = self._resolve_session_defaults(cursor)

            for (catalog, schema), group in sorted(
                by_dataset.items(),
                key=lambda kv: (kv[0][0] or "") + "/" + (kv[0][1] or ""),
            ):
                effective_project = catalog or current_proj
                effective_dataset = schema or current_ds
                if not effective_project or not effective_dataset:
                    for table in sorted(group, key=lambda t: t.render()):
                        h.update(b"no-dataset:")
                        h.update(table.render().encode())
                        h.update(b"\x00")
                    continue

                if not _is_valid_project_id(effective_project) or not _is_valid_dataset_id(
                    effective_dataset
                ):
                    raise RuntimeError(
                        f"BigQuery identifier {effective_project}.{effective_dataset!r} "
                        "is not valid"
                    )

                query = (
                    f"SELECT column_name, data_type, is_nullable "
                    f"FROM `{effective_project}.{effective_dataset}.INFORMATION_SCHEMA.COLUMNS` "
                    f"WHERE table_name = @table_name "
                    f"ORDER BY ordinal_position"
                )
                for table in sorted(group, key=lambda t: t.render()):
                    cursor.execute(query, parameters={"table_name": table.name})
                    rows = cursor.fetchall() or []
                    h.update(effective_project.encode())
                    h.update(b".")
                    h.update(effective_dataset.encode())
                    h.update(b".")
                    h.update(table.name.encode())
                    h.update(b":")
                    for col_name, data_type, is_nullable in rows:
                        h.update(str(col_name).encode())
                        h.update(b":")
                        h.update(str(data_type).encode())
                        h.update(b":")
                        h.update(str(is_nullable).encode())
                        h.update(b"\x00")
                    h.update(b"\x00")

        return SchemaFingerprint(value=h.digest())

    def list_schema(self, conn: Any) -> list[TableSchema]:
        """Enumerate tables and views in the spec's ``(project_id, dataset_id)`` only."""
        with conn.cursor() as cursor:
            project, dataset = self._resolve_session_defaults(cursor)
            if not project or not dataset:
                return []

            if not _is_valid_project_id(project) or not _is_valid_dataset_id(dataset):
                return []

            query = (
                f"SELECT t.table_catalog, t.table_schema, t.table_name, "
                f"       c.column_name, c.data_type, c.is_nullable "
                f"FROM `{project}.{dataset}.INFORMATION_SCHEMA.TABLES` t "
                f"JOIN `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS` c "
                f"     ON c.table_catalog = t.table_catalog "
                f"    AND c.table_schema  = t.table_schema "
                f"    AND c.table_name    = t.table_name "
                f"WHERE t.table_type IN ('BASE TABLE', 'VIEW') "
                f"ORDER BY t.table_schema, t.table_name, c.ordinal_position"
            )
            cursor.execute(query)
            rows = cursor.fetchall() or []

        grouped: dict[tuple[str | None, str | None, str], list[ColumnInfo]] = {}
        order: list[tuple[str | None, str | None, str]] = []
        for cat, sch, name, col_name, data_type, nullable_str in rows:
            key = (cat or None, sch or None, str(name))
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(
                ColumnInfo(
                    name=str(col_name),
                    type=str(data_type),
                    nullable=(str(nullable_str).upper() == "YES"),
                )
            )

        return [
            TableSchema(
                catalog=cat,
                schema=sch,
                name=name,
                columns=tuple(grouped[(cat, sch, name)]),
            )
            for (cat, sch, name) in order
        ]


_ADAPTER = BigQueryAdapter()


def register() -> None:
    """Register the adapter (idempotent); see drivers/__init__.py."""
    register_adapter(_ADAPTER)


register()
