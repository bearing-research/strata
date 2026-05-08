"""BigQuery driver adapter.

Backed by ``adbc-driver-bigquery``. The freshness probe reads
``last_modified_time`` from each touched dataset's per-dataset
``__TABLES__`` legacy view (BigQuery's
``INFORMATION_SCHEMA.TABLES`` doesn't expose a last-modified
column; ``__TABLES__`` is the documented fast path despite its
"legacy" label). The schema fingerprint walks
``INFORMATION_SCHEMA.COLUMNS``.

Read-only enforcement is **credentials-based**, not session-flag-
based — BigQuery has no equivalent of Postgres's
``SET default_transaction_read_only = on``. The security boundary
lives in the service account's IAM role grants. Read cells should
reference a connection whose ``credentials_path`` points at a
service account with ``roles/bigquery.dataViewer`` +
``roles/bigquery.jobUser``; write cells (``# @sql write=true``)
should reference ``write_credentials_path`` pointing at a service
account with ``roles/bigquery.dataEditor``. Without
``write_credentials_path``, write cells inherit the same
credentials as read cells.

**Streaming-buffer caveat.** Tables receiving streaming inserts
have ``last_modified_time`` lag by minutes-to-90-min until the
buffer flushes. Strata's freshness probe will *underestimate*
freshness for those tables — cells against streaming targets
should pin ``# @cache session`` to dodge the issue. Detecting
streaming buffers via ``tables.get → streamingBuffer`` is a
follow-up; v1 documents the limitation.

See ``docs/internal/design-sql-cells.md`` for the full design.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
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

_CAPABILITIES = AdapterCapabilities(
    per_table_freshness=True,
    # BigQuery time-travel exposes snapshot reads via ``FOR SYSTEM_TIME
    # AS OF``, but no per-table snapshot ID is exposed as a stable
    # equality token. Treat as equality-only.
    supports_snapshot=False,
    # ``__TABLES__`` and ``INFORMATION_SCHEMA`` aren't frozen inside
    # a transaction — the probe shares the query connection.
    needs_separate_probe_conn=False,
)

# BigQuery project IDs and dataset IDs follow different rules.
# Per the GCP docs:
#   - Project ID: lowercase letters / digits / hyphens, must start
#     with a letter (GCP enforces 6-30 chars but we accept any
#     length — the adapter's regex is for splice-injection
#     defense, not for length validation).
#   - Dataset ID: letters / digits / underscores, must start with
#     a letter or underscore.
# Splice-validated separately so we accept legitimate values from
# both columns of the ``project.dataset`` path while still rejecting
# the obvious injection shapes (semicolons, backticks, spaces).
_PROJECT_ID_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_DATASET_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_valid_project_id(value: str) -> bool:
    return bool(_PROJECT_ID_RE.match(value))


def _is_valid_dataset_id(value: str) -> bool:
    return bool(_DATASET_ID_RE.match(value))


def _spec_attr(spec: Any, key: str) -> Any:
    """Read a top-level field off a ``ConnectionSpec`` safely.

    Same Pydantic-bound-method workaround as the Snowflake adapter:
    ``BaseModel.schema`` shadows extras, so prefer ``model_extra``
    and reject bound-method shadows.
    """
    extras = getattr(spec, "model_extra", None) or {}
    if key in extras:
        return extras.get(key)
    value = getattr(spec, key, None)
    if callable(value) and getattr(value, "__self__", None) is not None:
        return None
    return value


def _credentials_principal(path_value: Any) -> str | None:
    """Read ``client_email`` from a service-account JSON file.

    The principal (the SA's email) is what BigQuery actually keys
    visibility on — different SA files for the same role-set give
    the same view, while different SAs give different views.
    Including the principal in the cache identity makes "swap to
    a different SA, see different objects" invalidate the cache
    correctly.

    Returns None when the file can't be read (e.g. relative path
    that hasn't been resolved yet, file moved between machines).
    The path itself still folds into the identity below as a
    fallback.
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
        # Test seam: pass a fake callable that takes the kwargs
        # dict the production code would have handed to ADBC.
        # Production lazy-imports the driver inside ``open()``.
        self._connect_fn = connect_fn

    # --- identity ---------------------------------------------------------

    def canonicalize_connection_id(self, spec: Any, *, read_only: bool = True) -> str:
        """Hash identity-shaping fields, excluding secrets.

        Identity-shaping for BigQuery: project_id, dataset_id, and
        the service account's principal (extracted from the
        credentials JSON when readable). ``read_only`` decides
        which credentials principal joins the hash — read cells
        fold only ``credentials_path``'s principal; write cells
        fold ``write_credentials_path``'s principal (falling back
        to ``credentials_path``). Without the read_only branch,
        swapping the write SA would churn read-cell caches even
        though read execution never uses it.

        When neither credentials path is configured, the adapter
        falls back to ambient Application Default Credentials at
        execute time. The principal is unknown at canonicalize
        time without a network call, so the identity carries an
        ``ambient_adc`` sentinel — two notebooks with no creds
        configured stay segregated from notebooks with explicit
        creds, but the cache may leak across machines that share
        a notebook spec but use different ambient principals.
        Users who want stable cache identity across machines
        should set an explicit ``credentials_path``.

        Excluded: the credentials file's *contents* (the signing
        key); paths fall back when principal extraction fails.
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

        # Read-side credentials always join. Write-side joins only
        # when ``read_only=False``. This way changing
        # ``write_credentials_path`` doesn't invalidate read-cell
        # caches that never touch the write principal.
        ro_path = _spec_attr(spec, "credentials_path")
        if ro_path:
            principal = _credentials_principal(ro_path)
            identity["credentials_principal"] = principal or str(ro_path)

        if not read_only:
            rw_path = _spec_attr(spec, "write_credentials_path")
            if rw_path:
                principal = _credentials_principal(rw_path)
                identity["write_credentials_principal"] = principal or str(rw_path)

        # When no credentials are configured at all (for the
        # active read/write side), the driver will pick up
        # ambient ADC at execute time. We can't know which
        # principal that is without a network call, so flag it
        # explicitly. Notebooks that want machine-portable cache
        # identity should configure ``credentials_path``.
        active_creds = (
            ro_path if read_only else (_spec_attr(spec, "write_credentials_path") or ro_path)
        )
        if not active_creds:
            identity["ambient_adc"] = True

        return identity

    # --- connection lifecycle --------------------------------------------

    def open(self, spec: Any, *, read_only: bool) -> Any:
        """Open an ADBC BigQuery connection.

        Read-only enforcement is **credentials-based**: if
        ``write_credentials_path`` is set, ``open(read_only=False)``
        uses it; otherwise both read and write paths use
        ``credentials_path``. The user's IAM role grants on the
        service account decide whether DML is permitted —
        BigQuery has no session-level read-only flag.

        The ADBC BigQuery driver doesn't take a URI; it takes
        keyword arguments (``adbc.bigquery.sql.project_id``,
        ``adbc.bigquery.sql.auth_credentials``, etc.). The
        connect_fn test seam mirrors this with a dict.
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
        """Read the session's effective project + dataset.

        BigQuery exposes ``@@dataset_id`` and ``@@project_id`` as
        session variables; both are session-scoped reads. Used to
        resolve unqualified tables in probes against whatever the
        connection's current default is — same idea as Snowflake's
        ``CURRENT_SCHEMA()`` resolution, just driver-specific.
        Returns ``(project, dataset)``; either may be None when
        the connection has no default set.
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
            # Not all BigQuery contexts expose @@dataset_id (e.g.
            # earlier ADBC versions, or BigQuery Omni). Fall back
            # to "no default" rather than crash the probe.
            return None, None

    # --- probes ----------------------------------------------------------

    def probe_freshness(
        self,
        probe_conn: Any,
        tables: list[QualifiedTable],
    ) -> FreshnessToken:
        """Per-table freshness via ``__TABLES__.last_modified_time``.

        ``__TABLES__`` is the legacy-but-stable per-dataset view
        that exposes ``last_modified_time`` (a unix-millis
        timestamp); ``INFORMATION_SCHEMA.TABLES`` doesn't. Tables
        are grouped by (project, dataset) so each touched dataset
        gets one ``__TABLES__`` query.

        Streaming-buffer caveat (see module docstring): tables
        receiving streaming inserts have lag here. Documented;
        users should pin ``# @cache session`` for those.
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
                key=lambda kv: ((kv[0][0] or "") + "/" + (kv[0][1] or "")),
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

        Catches schema evolution that ``last_modified_time``
        already covers (BigQuery DDL touches the timestamp); the
        explicit fingerprint is belt-and-suspenders.
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
                key=lambda kv: ((kv[0][0] or "") + "/" + (kv[0][1] or "")),
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
        """Enumerate tables and views in the connection's default dataset.

        Scopes to ``(project_id, dataset_id)`` from the spec — schema
        discovery across multiple datasets would mean one query per
        dataset, which we keep tight for v1.
        """
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
    """Idempotent registration entry point — see drivers/__init__.py."""
    register_adapter(_ADAPTER)


register()
