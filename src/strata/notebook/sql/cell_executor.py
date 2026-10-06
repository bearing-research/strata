"""SQL cell executor: the entry point ``CellExecutor`` calls for SQL cells.

Resolves the connection and ``# @cache`` policy, binds upstream variables,
checks the artifact store by SQL provenance, and on a miss runs the query on an
enforced read-only ADBC connection and stores an ``arrow/ipc`` artifact. Every
failure surfaces as the result dict's ``error``, as in the prompt executor.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import io
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, cast

from strata.notebook.annotations import parse_annotations
from strata.notebook.credentials import CredentialError, CredentialResolver, credential_identity
from strata.notebook.provenance import derive_subkey
from strata.notebook.serializer import _META_SHAPE, _SHAPE_SCALAR, _extract_scalar_from_table
from strata.notebook.sql.adapter import FreshnessToken
from strata.notebook.sql.analyzer import (
    analyze_sql_cell,
    confined_write_violation,
    read_only_violation,
    rewrite_named_to_positional,
)
from strata.notebook.sql.bind import BindError, resolve_bind_params
from strata.notebook.sql.lake import (
    PARAM_SNAPSHOT_IDS,
    Lake,
    LakeError,
    lake_options,
    local_database_problem,
    pin_snapshots,
    resolve_lake,
    snapshot_problem,
    snapshot_rows,
)
from strata.notebook.sql.provenance import (
    CachePolicyError,
    compute_sql_provenance_hash,
    normalize_query,
    resolve_cache_policy,
)
from strata.notebook.sql.registry import get_adapter
from strata.notebook.sql.time_travel import (
    PARAM_AT,
    PARAM_BASIS,
    PARAM_VALID_UNTIL,
    SnapshotPin,
    TimeTravelAdapter,
    supports_time_travel,
)

if TYPE_CHECKING:
    from strata.notebook.models import ConnectionSpec
    from strata.notebook.session import NotebookSession
    from strata.notebook.sql.adapter import DriverAdapter

logger = logging.getLogger(__name__)


async def execute_sql_cell(
    session: NotebookSession,
    cell_id: str,
    source: str,
    *,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Execute a SQL cell; returns the same dict shape as ``execute_prompt_cell``.

    Keys: ``success``, ``outputs``, ``stdout``, ``stderr``, ``error``,
    ``cache_hit``, ``duration_ms``, ``execution_method``, ``artifact_uri``.
    """
    start_time = time.time()

    # ---- annotations + connection resolution ----------------------
    annotations = parse_annotations(source)
    # Above every @cache policy, forever included.
    if annotations.nocache:
        use_cache = False
    if annotations.sql is None or not annotations.sql.connection:
        return _error_result(
            "SQL cell is missing `# @sql connection=<name>`.",
            start_time,
        )
    spec = _find_connection(session, annotations.sql.connection)
    if spec is None:
        return _error_result(
            f"unknown connection {annotations.sql.connection!r}; "
            "declare it under [connections.<name>] in notebook.toml",
            start_time,
        )

    try:
        adapter = get_adapter(spec.driver)
    except KeyError as exc:
        return _error_result(str(exc), start_time)
    problem = database_problem(spec, session.path, session._lake_config())
    if problem is not None:
        return _error_result(f"connection {spec.name!r}: {problem}", start_time)

    # Write cells: no freshness probe (the key is source-derived) and no
    # read-only enforcement.
    if annotations.sql.write:
        return await _execute_write_cell(
            session=session,
            cell_id=cell_id,
            source=source,
            adapter=adapter,
            spec=spec,
            annotations=annotations,
            start_time=start_time,
            use_cache=use_cache,
        )

    # ---- analyze ---------------------------------------------------
    analysis = analyze_sql_cell(source, dialect=adapter.sqlglot_dialect)
    if analysis.parse_error:
        return _error_result(f"SQL parse error: {analysis.parse_error}", start_time)
    if not analysis.sql_body:
        return _error_result("SQL cell body is empty.", start_time)
    # A body can end the driver's read-only transaction, so check before sending.
    violation = read_only_violation(
        rewrite_named_to_positional(analysis.sql_body, adapter.sqlglot_dialect),
        adapter.sqlglot_dialect,
    )
    if violation is not None:
        return _error_result(violation, start_time)

    # ---- bind params -----------------------------------------------
    namespace, upstream_input_hashes, input_refs = _load_upstream_variables(
        session, cell_id, analysis.references
    )
    try:
        params = resolve_bind_params(analysis.placeholder_positions, namespace)
    except BindError as exc:
        return _error_result(str(exc), start_time)

    # ---- cache policy ----------------------------------------------
    # A catalog's tables have true snapshot ids, so a lake connection can pin them.
    catalog, _ = lake_options(spec)
    capabilities = adapter.capabilities
    if catalog:
        capabilities = dataclasses.replace(capabilities, supports_snapshot=True)
    try:
        policy = resolve_cache_policy(
            analysis.cache_policy,
            capabilities=capabilities,
            session_id=session.id,
        )
    except CachePolicyError as exc:
        return _error_result(str(exc), start_time)

    # The fingerprint covers only tables the analyzer could name; one named at
    # run time could change unseen, so run the query. Explicit session, ttl or
    # forever policies are taken at their word.
    runs_every_time = policy.kind == "fingerprint" and bool(analysis.unresolved_tables)
    if runs_every_time:
        use_cache = False

    # The on-disk spec keeps relative paths so notebook.toml round-trips;
    # resolve them only for the adapter.
    try:
        runtime_spec = _resolve_runtime_spec(
            spec, session.path, _credentials(session), _auth_env(session)
        )
    except CredentialError as exc:
        return _error_result(f"connection {spec.name!r}: {exc}", start_time)

    query_normalized = normalize_query(analysis.sql_body, adapter.sqlglot_dialect)
    connection_id = _with_credential(
        adapter.canonicalize_connection_id(runtime_spec, read_only=True), spec
    )

    artifact_mgr = session.get_artifact_manager()
    notebook_id = session.notebook_state.id
    output_name = analysis.name
    canonical_id = f"nb_{notebook_id}_cell_{cell_id}_var_{output_name}"
    basis: str | None = None
    if policy.snapshot_required:
        # The query's identity minus the state it reads, to reuse a prior run's
        # timestamp or snapshots: that reuse is what makes it a snapshot.
        basis = compute_sql_provenance_hash(
            query_normalized=query_normalized,
            bind_params=params,
            connection_id=connection_id,
            upstream_input_hashes=upstream_input_hashes,
            cache_salt=policy.salt,
            freshness_token=None,
            schema_fingerprint=None,
        )
    pins_snapshots = basis is not None and bool(catalog)
    pinned = None
    if pins_snapshots:
        assert catalog is not None and basis is not None
        problem = snapshot_problem(catalog, analysis)
        if problem is not None:
            return _error_result(problem, start_time)
        pinned = _previous_snapshots(artifact_mgr, canonical_id, basis) if use_cache else None

    # ---- the lake: catalog tables' snapshots, mounts ---------------
    lake = None
    if any(lake_options(spec)):
        try:
            lake = resolve_lake(
                session, cell_id, source, runtime_spec, analysis.tables, pinned=pinned
            )
        except LakeError as exc:
            return _error_result(f"connection {spec.name!r}: {exc}", start_time)
        runtime_spec = lake.spec
    try:
        runtime_spec = _confined(session, runtime_spec, lake)
    except LakeError as exc:
        return _error_result(f"connection {spec.name!r}: {exc}", start_time)
    pinned_lake = lake if pins_snapshots else None

    # ---- probes (optional) -----------------------------------------
    freshness = None
    schema_fp = None
    pin: SnapshotPin | None = None
    if pinned_lake is not None:
        # The snapshot ids are the state; an Iceberg snapshot fixes its schema too.
        freshness = FreshnessToken(
            value=json.dumps(snapshot_rows(pinned_lake.snapshots)).encode(), is_snapshot=True
        )
    elif policy.snapshot_required and supports_time_travel(adapter) and basis is not None:
        pin = _previous_pin(artifact_mgr, canonical_id, basis) if use_cache else None
        if pin is None:
            try:
                pin = _take_pin(adapter, runtime_spec, analysis.tables)
            except Exception as exc:  # noqa: BLE001
                return _error_result(f"snapshot probe failed: {exc}", start_time)
        freshness = FreshnessToken(value=f"at:{pin.at}".encode(), is_snapshot=True)
    elif policy.freshness_required or policy.schema_required:
        try:
            freshness, schema_fp = await asyncio.to_thread(
                _run_probes, adapter, runtime_spec, analysis.tables, policy
            )
        except Exception as exc:  # noqa: BLE001
            return _error_result(f"probe failed: {exc}", start_time)
        if policy.snapshot_required and (freshness is None or not freshness.is_snapshot):
            return _error_result(
                "@cache snapshot requires a durable snapshot ID; "
                "the freshness probe returned an equality token instead",
                start_time,
            )

    # ---- provenance hash -------------------------------------------
    # Runtime spec, so a relative credential-file path can be read.
    provenance_hash = compute_sql_provenance_hash(
        query_normalized=query_normalized,
        bind_params=params,
        connection_id=connection_id,
        upstream_input_hashes=upstream_input_hashes,
        cache_salt=policy.salt,
        freshness_token=freshness,
        schema_fingerprint=schema_fp,
        lake_fingerprints=lake.fingerprints if lake else (),
    )
    var_provenance = derive_subkey(provenance_hash, output_name)

    # ---- cache check -----------------------------------------------

    if use_cache:
        cached = artifact_mgr.find_cached(var_provenance)
        if cached is not None:
            canonical = artifact_mgr.artifact_store.get_latest_version(canonical_id)
            if canonical is not None and canonical.provenance_hash == var_provenance:
                hit = _cache_hit_result(
                    artifact_mgr,
                    canonical,
                    output_name,
                    start_time,
                    session=session,
                    cell_id=cell_id,
                )
                return _report_snapshots(_report_pin(hit, pin), pinned_lake)

    # ---- execute query ---------------------------------------------
    try:
        table = await asyncio.to_thread(
            _execute_query,
            adapter,
            runtime_spec,
            analysis,
            params,
            at=pin.at if pin is not None else None,
            lake=lake,
        )
    except Exception as exc:  # noqa: BLE001
        return _error_result(
            f"SQL execution failed: {_exception_message(exc)}",
            start_time,
        )

    blob = _serialize_arrow_ipc(table)
    if runs_every_time:
        # The hash can't see the unnamed table change, so fold in the rows:
        # consumers re-run exactly when the data changed.
        provenance_hash = derive_subkey(
            provenance_hash, f"content={hashlib.sha256(blob).hexdigest()}"
        )
        var_provenance = derive_subkey(provenance_hash, output_name)
    artifact = artifact_mgr.store_cell_output(
        cell_id=cell_id,
        variable_name=output_name,
        blob_data=blob,
        content_type="arrow/ipc",
        provenance_hash=var_provenance,
        source_hash=provenance_hash,  # cell-level provenance for staleness
        source=source,
        # Bound variables, so lineage continues past the query.
        input_versions=input_refs,
        extra_params=_snapshot_params(basis, pin, pinned_lake),
    )
    uri = f"strata://artifact/{artifact.id}@v={artifact.version}"

    # Downstream provenance and input loading read ``artifact_uris``; without
    # this a consumer's hash would silently omit the SQL input.
    cell_state = next(
        (c for c in session.notebook_state.cells if c.id == cell_id),
        None,
    )
    if cell_state is not None:
        cell_state.artifact_uris[output_name] = uri
        cell_state.artifact_uri = uri

    duration_ms = (time.time() - start_time) * 1000
    # Query results can be huge: keep the default cap.
    display_output = _table_display(table)
    result = _report_pin(
        {
            "success": True,
            "outputs": {
                output_name: {
                    "content_type": "arrow/ipc",
                    "bytes": len(blob),
                    "preview": display_output["preview"],
                }
            },
            "display_outputs": [display_output],
            "display_output": display_output,
            "stdout": "",
            "stderr": "",
            "error": None,
            "cache_hit": False,
            "duration_ms": int(duration_ms),
            "execution_method": "sql",
            "artifact_uri": uri,
            "mutation_warnings": [],
        },
        pin,
    )
    return _report_snapshots(result, pinned_lake)


# --- helpers --------------------------------------------------------------


def _find_connection(session: NotebookSession, name: str) -> ConnectionSpec | None:
    for spec in session.notebook_state.connections:
        if spec.name == name:
            return spec
    return None


async def _execute_write_cell(
    *,
    session: NotebookSession,
    cell_id: str,
    source: str,
    adapter: DriverAdapter,
    spec: ConnectionSpec,
    annotations: Any,
    start_time: float,
    use_cache: bool,
) -> dict[str, Any]:
    """Run a ``# @sql connection=... write=true`` cell.

    Unlike reads: the connection is writable, the body may be multi-statement
    (run one by one), and the cache key is source-only with default policy
    ``session``. The artifact has one row per statement: ``stmt`` (1-indexed),
    ``kind`` and nullable ``rows_affected``.
    """
    from strata.notebook.annotations import CachePolicy

    # Same bind/placeholder surface as reads; only execution differs.
    analysis = analyze_sql_cell(source, dialect=adapter.sqlglot_dialect)
    if analysis.parse_error:
        return _error_result(f"SQL parse error: {analysis.parse_error}", start_time)
    if not analysis.sql_body:
        return _error_result("Write SQL cell body is empty.", start_time)

    # Probe-based policies make no sense for writes; say so.
    cache_annotation = annotations.cache or CachePolicy(kind="session")
    if cache_annotation.kind in {"fingerprint", "snapshot"}:
        return _error_result(
            f"@cache {cache_annotation.kind} isn't valid on a write cell — "
            "writes mutate state, so probe-based invalidation has no anchor. "
            "Use `# @cache session` (run once per session) or `# @cache forever` "
            "(idempotent setup; cache by source).",
            start_time,
        )

    try:
        policy = resolve_cache_policy(
            cache_annotation,
            capabilities=adapter.capabilities,
            session_id=session.id,
        )
    except CachePolicyError as exc:
        return _error_result(str(exc), start_time)

    # Upstream hashes in provenance, so a changed input invalidates the cache.
    namespace, upstream_input_hashes, input_refs = _load_upstream_variables(
        session, cell_id, analysis.references
    )
    try:
        params = resolve_bind_params(analysis.placeholder_positions, namespace)
    except BindError as exc:
        return _error_result(str(exc), start_time)

    # No probe slots. Runtime spec for relative credential paths;
    # ``read_only=False`` puts the write principal in the identity.
    try:
        runtime_spec = _resolve_runtime_spec(
            spec, session.path, _credentials(session), _auth_env(session)
        )
    except CredentialError as exc:
        return _error_result(f"connection {spec.name!r}: {exc}", start_time)
    try:
        runtime_spec = _confined(session, runtime_spec, None)
    except LakeError as exc:
        return _error_result(f"connection {spec.name!r}: {exc}", start_time)
    if spec.driver == "sqlite" and getattr(runtime_spec, "confine_to", None) is not None:
        violation = confined_write_violation(
            rewrite_named_to_positional(analysis.sql_body, adapter.sqlglot_dialect),
            adapter.sqlglot_dialect,
        )
        if violation is not None:
            return _error_result(f"connection {spec.name!r}: {violation}", start_time)
    query_normalized = normalize_query(analysis.sql_body, adapter.sqlglot_dialect)
    connection_id = _with_credential(
        adapter.canonicalize_connection_id(runtime_spec, read_only=False), spec
    )
    provenance_hash = compute_sql_provenance_hash(
        query_normalized=query_normalized,
        bind_params=params,
        connection_id=connection_id,
        upstream_input_hashes=upstream_input_hashes,
        cache_salt=policy.salt,
        freshness_token=None,
        schema_fingerprint=None,
    )
    output_name = analysis.name
    var_provenance = derive_subkey(provenance_hash, output_name)

    artifact_mgr = session.get_artifact_manager()
    notebook_id = session.notebook_state.id
    canonical_id = f"nb_{notebook_id}_cell_{cell_id}_var_{output_name}"

    if use_cache:
        cached = artifact_mgr.find_cached(var_provenance)
        if cached is not None:
            canonical = artifact_mgr.artifact_store.get_latest_version(canonical_id)
            if canonical is not None and canonical.provenance_hash == var_provenance:
                return _cache_hit_result(
                    artifact_mgr,
                    canonical,
                    output_name,
                    start_time,
                    session=session,
                    cell_id=cell_id,
                )

    try:
        stats = await asyncio.to_thread(
            _execute_write_statements, adapter, runtime_spec, analysis.sql_body, namespace
        )
    except Exception as exc:  # noqa: BLE001
        return _error_result(f"SQL execution failed: {_exception_message(exc)}", start_time)

    table = _synthesize_write_result_table(stats)
    blob = _serialize_arrow_ipc(table)
    # One status row per statement: truncating them is unhelpful.
    write_display_cap = max(20, table.num_rows)
    artifact = artifact_mgr.store_cell_output(
        cell_id=cell_id,
        variable_name=output_name,
        blob_data=blob,
        content_type="arrow/ipc",
        provenance_hash=var_provenance,
        source_hash=provenance_hash,
        source=source,
        input_versions=input_refs,
    )
    uri = f"strata://artifact/{artifact.id}@v={artifact.version}"

    cell_state = next(
        (c for c in session.notebook_state.cells if c.id == cell_id),
        None,
    )
    if cell_state is not None:
        cell_state.artifact_uris[output_name] = uri
        cell_state.artifact_uri = uri

    duration_ms = (time.time() - start_time) * 1000
    display_output = _table_display(table, max_rows=write_display_cap)
    return {
        "success": True,
        "outputs": {
            output_name: {
                "content_type": "arrow/ipc",
                "bytes": len(blob),
                "preview": display_output["preview"],
            }
        },
        "display_outputs": [display_output],
        "display_output": display_output,
        "stdout": "",
        "stderr": "",
        "error": None,
        "cache_hit": False,
        "duration_ms": int(duration_ms),
        "execution_method": "sql",
        "artifact_uri": uri,
        "mutation_warnings": [],
    }


def _fetch_arrow_table(cursor: Any) -> Any:
    """The result as an Arrow table, from a DuckDB or an ADBC cursor.

    DuckDB 1.5+ deprecates ``fetch_arrow_table`` for ``to_arrow_table``; ADBC
    has only ``fetch_arrow_table``.
    """
    to_arrow_table = getattr(cursor, "to_arrow_table", None)
    if to_arrow_table is not None:
        return to_arrow_table()
    return cursor.fetch_arrow_table()


def _split_statements(body: str, dialect: str) -> list[str] | None:
    """The body's statements, sliced out of the text the cell declares.

    Split on tokenizer semicolons, not regenerated from the parse tree: sqlglot
    does not round-trip every construct (it dropped a recursive CTE's column
    list), and a write cell must run what it declares. ``None`` means the body
    could not be tokenized (run it whole); ``[]`` means nothing to run.
    """
    import sqlglot
    from sqlglot.tokens import TokenType

    try:
        tokens = sqlglot.tokenize(body, read=dialect)
    except Exception:  # noqa: BLE001 - any tokenizer failure means "run it whole"
        return None

    # A fragment is a statement only if it holds a token: a trailing comment
    # sent as a statement makes the driver fail with "INTERNAL: (unknown error)".
    cuts = [token.start for token in tokens if token.token_type is TokenType.SEMICOLON]
    code = [token.start for token in tokens if token.token_type is not TokenType.SEMICOLON]

    # Both lists are in source order: one forward walk, not quadratic.
    statements: list[str] = []
    start = 0
    index = 0
    for cut in [*cuts, len(body)]:
        while index < len(code) and code[index] < start:
            index += 1
        if index < len(code) and code[index] < cut:
            statements.append(body[start:cut].strip())
        start = cut + 1
    return statements


def _execute_write_statements(
    adapter: DriverAdapter,
    spec: ConnectionSpec,
    body: str,
    namespace: dict[str, Any],
) -> dict[str, Any]:
    """Open writable and execute each statement with its own bind pass.

    Returns ``{"statements": [{"kind": str, "rows_affected": int | None}, ...]}``
    in source order. ``rows_affected`` is None when the driver reports no count
    (typically DDL); 0 is a real count. A failed statement aborts with partial
    state left in place. Commit failures propagate rather than reporting a
    success that persisted nothing.
    """
    import sqlglot

    from strata.notebook.sql.analyzer import _extract_placeholder_positions

    # Drop bare ``Semicolon`` nodes (trailing comments) as the split does, so
    # the lists align and each kind comes from the parse (text alone reads
    # ``WITH ... INSERT`` as "WITH", not DML).
    parsed = [
        statement
        for statement in sqlglot.parse(
            rewrite_named_to_positional(body, adapter.sqlglot_dialect),
            dialect=adapter.sqlglot_dialect,
        )
        if statement is not None and not isinstance(statement, sqlglot.exp.Semicolon)
    ]
    texts = _split_statements(body, adapter.sqlglot_dialect)
    if texts is None:
        # Untokenizable vendor syntax: run the body whole. Placeholders still
        # come from the regex path.
        prepared = [(body, _statement_kind_from_text(body))]
    elif len(texts) == len(parsed):
        # Run the text as written; the parse only gives the kind.
        prepared = list(zip(texts, (_statement_kind_from_expr(stmt) for stmt in parsed)))
    elif texts:
        # Not parsed one-to-one: infer the kind from the text.
        prepared = [(text, _statement_kind_from_text(text)) for text in texts]
    else:
        # Only comments: send nothing (the driver errors on a bare comment).
        prepared = []

    statements: list[dict[str, Any]] = []
    conn = adapter.open(spec, read_only=False)
    try:
        for stmt_text, stmt_kind in prepared:
            placeholders = _extract_placeholder_positions(stmt_text)
            if placeholders:
                stmt_params = resolve_bind_params(placeholders, namespace)
                stmt_to_execute = rewrite_named_to_positional(stmt_text, adapter.sqlglot_dialect)
            else:
                stmt_params = ()
                stmt_to_execute = stmt_text
            cursor = conn.cursor()
            try:
                if stmt_params:
                    cursor.execute(stmt_to_execute, parameters=stmt_params)
                else:
                    cursor.execute(stmt_to_execute)
                # PEP 249: -1 means unavailable, 0 is a real count. DML only:
                # SQLite's ``changes()`` after DDL reports the prior DML's count.
                rows_affected: int | None
                if _is_dml_kind(stmt_kind):
                    rc = getattr(cursor, "rowcount", -1)
                    if isinstance(rc, int) and rc >= 0:
                        rows_affected = rc
                    elif getattr(adapter, "name", None) == "sqlite":
                        # ADBC SQLite never populates cursor.rowcount.
                        rows_affected = _sqlite_last_changes(conn)
                    else:
                        rows_affected = None
                else:
                    rows_affected = None
            finally:
                _safely_close(cursor)
            statements.append({"kind": stmt_kind, "rows_affected": rows_affected})

        # ADBC defaults to autocommit=False; a commit error fails the cell.
        commit = getattr(conn, "commit", None)
        if callable(commit):
            commit()
    finally:
        _safely_close(conn)

    return {"statements": statements}


_DML_KINDS = frozenset({"INSERT", "UPDATE", "DELETE", "MERGE", "REPLACE"})


def _is_dml_kind(kind: str) -> bool:
    """``rows_affected`` only applies to DML; DDL is null on display."""
    if not kind:
        return False
    head = kind.split()[0].upper()
    return head in _DML_KINDS


def _sqlite_last_changes(conn: Any) -> int | None:
    """Recover the last DML's row count via ``SELECT changes()``.

    ADBC's SQLite driver always reports ``rowcount`` -1. Errors return None.
    """
    try:
        cur = conn.cursor()
        try:
            cur.execute("SELECT changes()")
            tbl = _fetch_arrow_table(cur)
            rows = tbl.to_pylist()
            if rows:
                # Column is named "changes()"; read by value in case that changes.
                for v in rows[0].values():
                    if isinstance(v, int) and v >= 0:
                        return v
        finally:
            _safely_close(cur)
    except Exception:  # noqa: BLE001
        logger.exception("sqlite changes() probe failed")
    return None


def _statement_kind_from_expr(expr: Any) -> str:
    """Best-effort ``kind`` label for a parsed statement, e.g. ``CREATE TABLE`` or ``INSERT``."""
    if expr is None:
        return "UNKNOWN"
    cls_name = type(expr).__name__.upper()
    if cls_name in {"CREATE", "DROP"}:
        kind_arg = expr.args.get("kind") if hasattr(expr, "args") else None
        kind_str = (kind_arg or "TABLE").upper() if isinstance(kind_arg, str) else "TABLE"
        return f"{cls_name} {kind_str}"
    if cls_name in {"ALTERTABLE", "ALTER"}:
        return "ALTER TABLE"
    return cls_name


def _statement_kind_from_text(text: str) -> str:
    """Fallback kind when sqlglot can't parse: the first keyword after comments, uppercased."""
    cleaned = text.lstrip()
    while cleaned.startswith("--"):
        nl = cleaned.find("\n")
        cleaned = cleaned[nl + 1 :].lstrip() if nl != -1 else ""
    head = cleaned.split(None, 2)
    if not head:
        return "UNKNOWN"
    first = head[0].upper().rstrip(";")
    if first in {"CREATE", "DROP", "ALTER"} and len(head) > 1:
        return f"{first} {head[1].upper().rstrip(';')}"
    return first or "UNKNOWN"


def _synthesize_write_result_table(stats: dict[str, Any]) -> Any:
    """Per-statement Arrow table for a write cell: ``stmt``, ``kind``, ``rows_affected``."""
    import pyarrow as pa

    statements = stats.get("statements") or []
    return pa.table(
        {
            "stmt": pa.array(
                list(range(1, len(statements) + 1)),
                type=pa.int32(),
            ),
            "kind": pa.array(
                [s.get("kind", "UNKNOWN") for s in statements],
                type=pa.string(),
            ),
            "rows_affected": pa.array(
                [s.get("rows_affected") for s in statements],
                type=pa.int64(),
            ),
        }
    )


def _credentials(session: NotebookSession) -> CredentialResolver:
    """Named credentials as this notebook sees them, secret-manager values included."""
    return CredentialResolver.from_config(
        session._lake_config(), env=dict(session.notebook_state.env)
    )


def sql_reopen_identity(cell: Any, session: Any) -> str | None:
    """What a reopened SQL cell's cached rows depend on beyond its query.

    The connection and cache policy, which the generic triplet does not see.
    ``None`` for ``fingerprint`` and ``snapshot`` policies, which need a query
    to the source. ``forever``, ``session`` and ``ttl`` settle here (this
    session's salt, this moment's ttl bucket).
    """
    annotations = parse_annotations(cell.source)
    if annotations.sql is None or not annotations.sql.connection:
        return None
    spec = _find_connection(session, annotations.sql.connection)
    if spec is None:
        return None
    try:
        adapter = get_adapter(spec.driver)
    except KeyError:
        return None
    analysis = analyze_sql_cell(cell.source, dialect=adapter.sqlglot_dialect)
    try:
        policy = resolve_cache_policy(
            analysis.cache_policy,
            capabilities=adapter.capabilities,
            session_id=session.id,
        )
    except CachePolicyError:
        return None
    if policy.freshness_required:
        return None
    try:
        # No credentials: resolving may hit a secret manager, and the identity
        # carries only the credential's name.
        runtime_spec = _resolve_runtime_spec(spec, session.path, auth_env=_auth_env(session))
        connection_id = _with_credential(
            adapter.canonicalize_connection_id(runtime_spec, read_only=True), spec
        )
    except (CredentialError, ValueError, OSError):
        return None
    return hashlib.sha256(connection_id.encode() + b"|" + policy.salt).hexdigest()


# The files the server opens for a connection, by driver.
_OPENED_FILES = {
    "duckdb": ("path",),
    "sqlite": ("path",),
    "bigquery": ("credentials_path", "write_credentials_path"),
}


def _confined(session: Any, spec: ConnectionSpec, lake: Any) -> ConnectionSpec:
    """In service mode, a connection confined to its own database and lake.

    SQL cells run in the server process. DuckDB is confined by
    ``duckdb._confine`` to its lake's locations; a SQLite write cell is refused
    statements that reach other files (``confined_write_violation``). The files
    it opens are opened by their resolved paths, checked again here, so a link
    changed since ``database_problem`` ran is not followed.

    Raises:
        LakeError: a file it opens is not one the notebook may read.
    """
    config = session._lake_config()
    if getattr(config, "deployment_mode", "personal") != "service":
        return spec
    update: dict[str, Any] = {}
    extras = spec.model_extra or {}
    for key in _OPENED_FILES.get(spec.driver, ()):
        value = extras.get(key, getattr(spec, key, None))
        if isinstance(value, str) and value and value != ":memory:":
            update[key] = os.path.realpath(value)
    if update:
        problem = database_problem(spec.model_copy(update=update), session.path, config)
        if problem is not None:
            raise LakeError(problem)
    if spec.driver in ("duckdb", "sqlite"):
        update["confine_to"] = list(lake.locations) if lake else []
    return spec.model_copy(update=update) if update else spec


def database_problem(spec: ConnectionSpec, notebook_dir: Any, config: Any) -> str | None:
    """Why a service-mode SQL cell may not open *spec*'s database file, or None.

    The file is opened by the server process, so it must be one the notebook
    may read: see ``local_database_problem``. A SQLite ``uri`` is refused, since
    its parameters can name any file. A BigQuery key file must be in the
    notebook's directory.
    """
    from pathlib import Path

    if getattr(config, "deployment_mode", "personal") != "service":
        return None
    if spec.driver == "bigquery":
        return _key_file_problem(spec, notebook_dir)
    if spec.driver not in ("duckdb", "sqlite"):
        return None
    if spec.driver == "sqlite" and getattr(spec, "uri", None):
        return "a SQLite `uri` is not allowed on this server; name the database file with `path`"
    path = getattr(spec, "path", None)
    if not isinstance(path, str) or not path or path == ":memory:":
        return None
    problem = local_database_problem(str(Path(str(notebook_dir)) / path), notebook_dir, config)
    if problem is None:
        return None
    return f"the database {path} is outside this notebook's directory and {problem}"


def _key_file_problem(spec: ConnectionSpec, notebook_dir: Any) -> str | None:
    """Why a service-mode BigQuery connection may not read its key files, or None."""
    from pathlib import Path

    own = Path(os.path.realpath(str(notebook_dir)))
    extras = spec.model_extra or {}
    for key in ("credentials_path", "write_credentials_path"):
        value = extras.get(key, getattr(spec, key, None))
        if not isinstance(value, str) or not value:
            continue
        if own not in Path(os.path.realpath(Path(str(notebook_dir)) / value)).parents:
            return (
                f"`{key}` {value} is outside this notebook's directory, "
                "and this server reads a key file only from there"
            )
    return None


def _with_credential(connection_id: str, spec: ConnectionSpec) -> str:
    """Fold the credential's name into the connection's identity.

    Different credentials can see different objects, so cache entries are not
    shared; values stay out, so rotating a secret keeps every entry.
    """
    identity = credential_identity(spec.credential)
    if not identity:
        return connection_id
    return hashlib.sha256(f"{connection_id}|{identity}".encode()).hexdigest()


def _resolve_runtime_spec(
    spec: ConnectionSpec,
    notebook_dir: Any,
    credentials: CredentialResolver | None = None,
    auth_env: dict[str, str] | None = None,
) -> ConnectionSpec:
    """Return a spec copy with relative file paths and the credential resolved.

    The credential is resolved into ``auth`` beneath the block's own entries,
    so adapters never see names; ``CredentialError`` names it on failure.
    Relative paths (and BigQuery's ``credentials_path`` /
    ``write_credentials_path``) are rebased on the notebook directory, since the
    server's CWD is unrelated. ``uri`` passes through as-is. With *auth_env*
    (see ``_auth_env``), ``${VAR}`` in ``auth`` is resolved from it here, so the
    driver never reads the server's environment.
    """
    from pathlib import Path

    nb_dir = Path(str(notebook_dir))

    def _rebase(value: Any) -> Any:
        if not isinstance(value, str) or not value:
            return value
        p = Path(value)
        if p.is_absolute():
            return value
        return str((nb_dir / p).resolve())

    update: dict[str, Any] = {}
    raw_path = getattr(spec, "path", None)
    new_path = _rebase(raw_path)
    if new_path != raw_path:
        update["path"] = new_path

    # Driver-specific paths may be Pydantic extras (BigQuery's credentials_path);
    # ConnectionSpec has ``extra='allow'``, so ``model_copy`` accepts them.
    extras = getattr(spec, "model_extra", None) or {}
    for key in ("credentials_path", "write_credentials_path"):
        raw_value = extras.get(key) if key in extras else getattr(spec, key, None)
        new_value = _rebase(raw_value)
        if new_value != raw_value:
            update[key] = new_value

    auth = dict(spec.auth)
    if auth_env is not None:
        for key, value in auth.items():
            if value.startswith("${") and value.endswith("}"):
                if value[2:-1] not in auth_env:
                    raise CredentialError(
                        f"auth.{key} references {value}, which this notebook's env does not set"
                    )
                auth[key] = auth_env[value[2:-1]]
        if auth != spec.auth:
            update["auth"] = auth
    if spec.credential:
        resolved = (credentials or CredentialResolver()).resolve(spec.credential)
        update["auth"] = {**resolved, **auth}

    if not update:
        return spec
    return spec.model_copy(update=update)


def _auth_env(session: Any) -> dict[str, str] | None:
    """What ``${VAR}`` in connection auth reads in service mode: the notebook's env.

    ``None`` (personal mode) leaves it to the driver, which reads the server's
    environment; on a shared server that would send it to a member-chosen host.
    """
    if getattr(session._lake_config(), "deployment_mode", "personal") != "service":
        return None
    return dict(session.notebook_state.env)


def _load_upstream_variables(
    session: NotebookSession,
    cell_id: str,
    references: list[str],
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    """Load upstream values, their artifact hashes, and their refs.

    Returns ``(namespace, upstream_input_hashes, input_refs)``. Hashes feed
    provenance. Refs (``{strata://artifact/<id>@v=<n>: <id>@v=<n>}``) feed
    lineage, since a hash names no artifact and a bound variable would
    otherwise be missing from the chain.
    """
    namespace: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    refs: dict[str, str] = {}
    cell = next((c for c in session.notebook_state.cells if c.id == cell_id), None)
    if cell is None:
        return namespace, hashes, refs

    artifact_mgr = session.get_artifact_manager()
    notebook_id = session.notebook_state.id
    references_set = set(references)

    for upstream_id in cell.upstream_ids:
        upstream_cell = next(
            (c for c in session.notebook_state.cells if c.id == upstream_id),
            None,
        )
        if upstream_cell is None:
            continue
        for var_name in upstream_cell.defines:
            if var_name not in references_set:
                continue
            canonical_id = f"nb_{notebook_id}_cell_{upstream_id}_var_{var_name}"
            artifact = artifact_mgr.artifact_store.get_latest_version(canonical_id)
            if artifact is None:
                continue
            hashes[var_name] = artifact.provenance_hash
            ref = f"{canonical_id}@v={artifact.version}"
            refs[f"strata://artifact/{ref}"] = ref
            blob = artifact_mgr.load_artifact_data(canonical_id, artifact.version)
            content_type = _content_type_of(artifact)
            namespace[var_name] = _deserialize_blob(blob, content_type)

    return namespace, hashes, refs


def _content_type_of(artifact: Any) -> str:
    spec = getattr(artifact, "transform_spec", None)
    if not spec:
        return "json/object"
    try:
        parsed = json.loads(spec)
        return parsed.get("params", {}).get("content_type", "json/object")
    except (ValueError, KeyError):
        return "json/object"


class PickledObject:
    """A pickled upstream value, left unloaded; its type name is what a ``BindError`` shows."""


def _deserialize_blob(blob: bytes, content_type: str) -> Any:
    """Decode just enough of an artifact for a SQL bind: the scalar and Arrow paths."""
    if content_type == "json/object":
        try:
            return json.loads(blob)
        except (ValueError, TypeError):
            return None
    if content_type == "arrow/ipc":
        # A table is not bindable, but is loaded so the bind layer raises the right BindError.
        try:
            import pyarrow as pa

            table = pa.ipc.open_stream(blob).read_all()
        except Exception:  # noqa: BLE001
            return blob
        # datetime / Decimal / UUID / bytes values are stored as 1-row scalar tables.
        if (table.schema.metadata or {}).get(_META_SHAPE) == _SHAPE_SCALAR:
            return _extract_scalar_from_table(table)
        return table
    if content_type == "pickle/object":
        # Unpickling would run the producing cell's code as the server user.
        return PickledObject()
    return blob


def _run_probes(
    adapter: DriverAdapter,
    spec: ConnectionSpec,
    tables: list[Any],
    policy: Any,
) -> tuple[Any, Any]:
    """Run freshness and schema probes per the resolved policy.

    Adapters with ``needs_separate_probe_conn`` (Postgres) get their own
    connection, since stats are frozen inside the query connection's transaction.
    """
    freshness = None
    schema_fp = None
    probe_conn = adapter.open(spec, read_only=True)
    try:
        if policy.freshness_required:
            freshness = adapter.probe_freshness(probe_conn, tables)
        if policy.schema_required:
            schema_fp = adapter.probe_schema(probe_conn, tables)
    finally:
        _safely_close(probe_conn)
    return freshness, schema_fp


def _previous_pin(artifact_mgr: Any, canonical_id: str, basis: str) -> SnapshotPin | None:
    """The timestamp the last run of this same query was pinned to, if any."""
    canonical = artifact_mgr.artifact_store.get_latest_version(canonical_id)
    if canonical is None or not canonical.transform_spec:
        return None
    try:
        params = json.loads(canonical.transform_spec).get("params") or {}
    except ValueError:
        return None
    if params.get(PARAM_BASIS) != basis or not params.get(PARAM_AT):
        return None
    return SnapshotPin(at=params[PARAM_AT], valid_until=params.get(PARAM_VALID_UNTIL))


def _previous_snapshots(
    artifact_mgr: Any, canonical_id: str, basis: str
) -> dict[tuple[str, str], int] | None:
    """The catalog snapshots the last run of this same query read, if any."""
    canonical = artifact_mgr.artifact_store.get_latest_version(canonical_id)
    if canonical is None or not canonical.transform_spec:
        return None
    try:
        params = json.loads(canonical.transform_spec).get("params") or {}
        rows = json.loads(params.get(PARAM_SNAPSHOT_IDS) or "null")
    except ValueError:
        return None
    if params.get(PARAM_BASIS) != basis or not rows:
        return None
    return {(namespace, name): int(snapshot) for namespace, name, snapshot in rows}


def _snapshot_params(
    basis: str | None, pin: SnapshotPin | None, lake: Lake | None
) -> dict[str, str] | None:
    """What a snapshot cell records on its artifact so a later run reads the same state."""
    if basis is None:
        return None
    if pin is not None:
        return {
            PARAM_BASIS: basis,
            PARAM_AT: pin.at,
            **({PARAM_VALID_UNTIL: pin.valid_until} if pin.valid_until else {}),
        }
    if lake is not None:
        return {PARAM_BASIS: basis, PARAM_SNAPSHOT_IDS: json.dumps(snapshot_rows(lake.snapshots))}
    return None


def _report_snapshots(result: dict[str, Any], lake: Lake | None) -> dict[str, Any]:
    """Say which catalog snapshots a snapshot cell shows."""
    if lake is None:
        return result
    catalog, _ = lake_options(lake.spec)
    pinned = ", ".join(
        f"{catalog}.{namespace}.{name} at snapshot {snapshot}"
        for namespace, name, snapshot in snapshot_rows(lake.snapshots)
    )
    result["stdout"] = (result.get("stdout") or "") + f"State of {pinned}.\n"
    return result


def _take_pin(adapter: Any, spec: ConnectionSpec, tables: list[Any]) -> SnapshotPin:
    """A new pin at the warehouse's current time, with its retention horizon."""
    conn = adapter.open(spec, read_only=True)
    try:
        at = adapter.snapshot_timestamp(conn)
        return SnapshotPin(at=at, valid_until=adapter.retention_until(conn, tables, at))
    finally:
        _safely_close(conn)


def _report_pin(result: dict[str, Any], pin: SnapshotPin | None) -> dict[str, Any]:
    """Say which moment a snapshot cell shows, and how long it stays queryable."""
    if pin is None:
        return result
    horizon = pin.valid_until or "unknown"
    result["stdout"] = (
        result.get("stdout") or ""
    ) + f"State as of {pin.at}; queryable until {horizon}.\n"
    result["snapshot_at"] = pin.at
    result["snapshot_valid_until"] = pin.valid_until
    return result


def _execute_query(
    adapter: DriverAdapter,
    spec: ConnectionSpec,
    analysis: Any,
    params: tuple[Any, ...],
    *,
    at: str | None = None,
    lake: Lake | None = None,
) -> Any:
    """Open a read-only connection, run the rewritten query, fetch Arrow.

    With *at*, every table is read as of that timestamp (a snapshot cell).
    """
    rewritten = rewrite_named_to_positional(analysis.sql_body, adapter.sqlglot_dialect)
    if at is not None:
        rewritten = cast("TimeTravelAdapter", adapter).pin_query(rewritten, at)
    if lake is not None:
        catalog, _ = lake_options(lake.spec)
        if catalog:
            rewritten = pin_snapshots(rewritten, catalog, lake.snapshots)
    conn = adapter.open(spec, read_only=True)
    try:
        cursor = conn.cursor()
        try:
            if params:
                cursor.execute(rewritten, parameters=params)
            else:
                cursor.execute(rewritten)
            return _fetch_arrow_table(cursor)
        finally:
            _safely_close(cursor)
    finally:
        _safely_close(conn)


def _exception_message(exc: BaseException) -> str:
    """Walk the exception chain so ADBC's wrapped errors stay visible.

    ADBC often raises a generic ``"INTERNAL: (unknown error)"`` with the useful
    message in a chained exception.
    """
    parts: list[str] = []
    seen: set[str] = set()
    cur: BaseException | None = exc
    while cur is not None:
        msg = str(cur).strip()
        if msg and msg not in seen:
            seen.add(msg)
            parts.append(msg)
        cur = cur.__cause__ or cur.__context__
    return " | ".join(parts) or type(exc).__name__


def _safely_close(handle: Any) -> None:
    if handle is None:
        return
    try:
        handle.close()
    except Exception:  # noqa: BLE001
        logger.exception("error closing handle")
        # Mark it closed anyway. adbc sets ``_closed`` only after
        # ``_stmt.close()`` returns, and a write on a read-only connection
        # raises there; ``__del__`` would then close it again, underflowing the
        # driver's child count as an unraisable error on some unrelated cell.
        if getattr(handle, "_closed", None) is False:
            handle._closed = True


def _serialize_arrow_ipc(table: Any) -> bytes:
    import pyarrow as pa

    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


_WRITE_STATUS_COLUMNS = ("stmt", "kind", "rows_affected")


def _table_display(table: Any, *, max_rows: int = 5) -> dict[str, Any]:
    """Build a small markdown preview for the cell's display panel.

    ``max_rows`` defaults to 5. Write-cell status tables (detected by their
    ``stmt, kind, rows_affected`` schema, so cache hits match too) are not
    truncated, since their rows are statements rather than data.
    """
    rows = table.num_rows
    cols = table.num_columns
    if tuple(table.schema.names) == _WRITE_STATUS_COLUMNS:
        max_rows = max(max_rows, rows)
    sample = min(rows, max_rows)
    head = table.slice(0, sample).to_pylist() if sample else []
    column_names = list(table.schema.names)

    preview_lines = [f"{rows} rows × {cols} cols"]
    if column_names:
        preview_lines.append("| " + " | ".join(column_names) + " |")
        preview_lines.append("| " + " | ".join("---" for _ in column_names) + " |")
        for row in head:
            preview_lines.append(
                "| " + " | ".join(_format_cell(row.get(c)) for c in column_names) + " |"
            )
        if rows > sample:
            preview_lines.append(f"… {rows - sample} more rows")
    preview = "\n".join(preview_lines)
    return {
        "content_type": "text/markdown",
        "preview": preview,
        "markdown_text": preview,
    }


def _format_cell(value: Any) -> str:
    """Render a value for the markdown preview; ``None`` shows as a dash, not ``None``."""
    if value is None:
        return "—"
    return str(value)


def _cache_hit_result(
    artifact_mgr: Any,
    canonical: Any,
    output_name: str,
    start_time: float,
    *,
    session: NotebookSession | None = None,
    cell_id: str | None = None,
) -> dict[str, Any]:
    blob = artifact_mgr.load_artifact_data(canonical.id, canonical.version)
    import pyarrow as pa

    table = pa.ipc.open_stream(blob).read_all()
    duration_ms = (time.time() - start_time) * 1000
    display_output = _table_display(table)
    uri = f"strata://artifact/{canonical.id}@v={canonical.version}"

    # As on a miss: a hit after reopen would otherwise leave artifact_uris
    # empty and downstream caches stale.
    if session is not None and cell_id is not None:
        cell_state = next(
            (c for c in session.notebook_state.cells if c.id == cell_id),
            None,
        )
        if cell_state is not None:
            cell_state.artifact_uris[output_name] = uri
            cell_state.artifact_uri = uri
    return {
        "success": True,
        "outputs": {
            output_name: {
                "content_type": "arrow/ipc",
                "bytes": len(blob),
                "preview": display_output["preview"],
            }
        },
        "display_outputs": [display_output],
        "display_output": display_output,
        "stdout": "",
        "stderr": "",
        "error": None,
        "cache_hit": True,
        "duration_ms": int(duration_ms),
        "execution_method": "cached",
        "artifact_uri": uri,
        "mutation_warnings": [],
    }


def _error_result(message: str, start_time: float) -> dict[str, Any]:
    duration_ms = (time.time() - start_time) * 1000
    return {
        "success": False,
        "outputs": {},
        "display_outputs": [],
        "display_output": None,
        "stdout": "",
        "stderr": "",
        "error": message,
        "cache_hit": False,
        "duration_ms": int(duration_ms),
        "execution_method": "sql",
        "artifact_uri": None,
        "mutation_warnings": [],
    }
