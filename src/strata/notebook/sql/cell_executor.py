"""SQL cell executor — coordinator for the language=='sql' path.

Mirrors the structure of ``prompt_executor.execute_prompt_cell``: a
single entry point ``execute_sql_cell`` that ``CellExecutor`` calls
when it dispatches a SQL cell. The function:

1. Parses annotations and resolves the connection / adapter.
2. Runs the analyzer with the adapter's dialect.
3. Loads upstream variables and resolves bind parameters.
4. Resolves the ``# @cache`` policy and runs probes if required.
5. Computes the SQL provenance hash via the helpers in
   ``strata.notebook.sql.provenance``.
6. Checks the artifact store for a cache hit; on hit, returns the
   cached Arrow Table.
7. On miss, opens an enforced read-only ADBC connection, executes
   the query with positional binds, and stores the result as an
   ``arrow/ipc`` artifact.

Failure modes (missing connection, unknown driver, parse error,
read-only-enforcement failure, cache-policy error, bind error) all
surface as a populated ``error`` field in the result dict — same
shape the prompt executor uses — so the caller can render them in
the cell's output panel without special handling.
"""

from __future__ import annotations

import io
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from strata.notebook.annotations import parse_annotations
from strata.notebook.provenance import derive_subkey
from strata.notebook.sql.analyzer import analyze_sql_cell, rewrite_named_to_positional
from strata.notebook.sql.bind import BindError, resolve_bind_params
from strata.notebook.sql.provenance import (
    CachePolicyError,
    compute_sql_provenance_hash,
    normalize_query,
    resolve_cache_policy,
)
from strata.notebook.sql.registry import get_adapter

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
    """Execute a SQL cell. Return shape mirrors ``execute_prompt_cell``.

    Returns a dict with ``success``, ``outputs``, ``stdout``,
    ``stderr``, ``error``, ``cache_hit``, ``duration_ms``,
    ``execution_method``, ``artifact_uri``. The notebook executor
    lifts these into a ``CellExecutionResult``.
    """
    start_time = time.time()

    # ---- annotations + connection resolution ----------------------
    annotations = parse_annotations(source)
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

    # Write cells take a separate path: open the connection
    # writable, split the body into individual statements, run each
    # one, return a synthetic status table. No freshness probe (the
    # cache key is purely source-derived for writes), no read-only
    # enforcement.
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

    # ---- bind params -----------------------------------------------
    namespace, upstream_input_hashes = _load_upstream_variables(
        session, cell_id, analysis.references
    )
    try:
        params = resolve_bind_params(analysis.placeholder_positions, namespace)
    except BindError as exc:
        return _error_result(str(exc), start_time)

    # ---- cache policy ----------------------------------------------
    try:
        policy = resolve_cache_policy(
            analysis.cache_policy,
            capabilities=adapter.capabilities,
            session_id=session.id,
        )
    except CachePolicyError as exc:
        return _error_result(str(exc), start_time)

    # The on-disk spec keeps relative paths verbatim (so notebook.toml
    # round-trips byte-for-byte); resolve them just before the
    # adapter sees them so the in-process call site stays
    # notebook-unaware.
    runtime_spec = _resolve_runtime_spec(spec, session.path)

    # ---- probes (optional) -----------------------------------------
    freshness = None
    schema_fp = None
    if policy.freshness_required or policy.schema_required:
        try:
            freshness, schema_fp = _run_probes(adapter, runtime_spec, analysis.tables, policy)
        except Exception as exc:  # noqa: BLE001
            return _error_result(f"probe failed: {exc}", start_time)
        if policy.snapshot_required and (freshness is None or not freshness.is_snapshot):
            return _error_result(
                "@cache snapshot requires a durable snapshot ID; "
                "the freshness probe returned an equality token instead",
                start_time,
            )

    # ---- provenance hash -------------------------------------------
    # The runtime spec is what gets handed to ``open()``, with
    # relative paths rebased against the notebook directory. Use
    # it for canonicalize too so credential-file principal
    # extraction can read the file when the path is relative on
    # disk.
    query_normalized = normalize_query(analysis.sql_body, adapter.sqlglot_dialect)
    connection_id = adapter.canonicalize_connection_id(runtime_spec, read_only=True)
    provenance_hash = compute_sql_provenance_hash(
        query_normalized=query_normalized,
        bind_params=params,
        connection_id=connection_id,
        upstream_input_hashes=upstream_input_hashes,
        cache_salt=policy.salt,
        freshness_token=freshness,
        schema_fingerprint=schema_fp,
    )
    output_name = analysis.name
    var_provenance = derive_subkey(provenance_hash, output_name)

    # ---- cache check -----------------------------------------------
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

    # ---- execute query ---------------------------------------------
    try:
        table = _execute_query(adapter, runtime_spec, analysis, params)
    except Exception as exc:  # noqa: BLE001
        return _error_result(
            f"SQL execution failed: {_exception_message(exc)}",
            start_time,
        )

    blob = _serialize_arrow_ipc(table)
    artifact = artifact_mgr.store_cell_output(
        cell_id=cell_id,
        variable_name=output_name,
        blob_data=blob,
        content_type="arrow/ipc",
        provenance_hash=var_provenance,
        source_hash=provenance_hash,  # cell-level provenance for staleness
    )
    uri = f"strata://artifact/{artifact.id}@v={artifact.version}"

    # Make the artifact discoverable from downstream cells via the
    # cell-state ``artifact_uris`` map. ``_collect_input_hashes`` and
    # ``_load_input_blobs`` both walk these on the upstream cell, so
    # without this, a downstream Python cell that consumes a SQL
    # cell's output would compute its provenance without the SQL
    # input hash — a silent staleness leak.
    cell_state = next(
        (c for c in session.notebook_state.cells if c.id == cell_id),
        None,
    )
    if cell_state is not None:
        cell_state.artifact_uris[output_name] = uri
        cell_state.artifact_uri = uri

    duration_ms = (time.time() - start_time) * 1000
    # Read path: query results can be huge, keep the default cap.
    # Write-path display passes its own larger cap below.
    display_output = _table_display(table)
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
    """Run a ``# @sql connection=… write=true`` cell.

    Differs from the read path in three ways:

    1. The adapter opens the connection writable. SQLite drops the
       ``mode=ro`` URI override and the ``PRAGMA query_only=ON``;
       Postgres skips ``SET default_transaction_read_only=on``.
    2. The body may be multi-statement. ADBC's cursor.execute()
       runs only the first statement on most drivers, so we split
       via sqlglot (dialect-aware) and execute each in sequence.
    3. The cache key is source-only — no freshness probe, no
       schema fingerprint. Default policy is ``session`` so a
       seed cell runs once per session and dedupes subsequent
       runs within the same session. Users opt into other policies
       via ``# @cache``.

    The cell still produces an Arrow artifact: one row per
    statement with ``stmt`` (1-indexed), ``kind`` (CREATE TABLE /
    INSERT / etc), and ``rows_affected`` (nullable; None when the
    driver doesn't report — typically DDL). Downstream cells with
    ``# @after seed`` find the artifact via ``cell.artifact_uris``.
    """
    from strata.notebook.annotations import CachePolicy

    # Run the analyzer so the write path participates in the same
    # bind / placeholder / # @name surface as the read path. The
    # only divergence below is execution: read calls
    # adapter.probe_freshness then runs the (single) statement
    # read-only; write opens writable, splits the body via
    # sqlglot, and runs each statement with its own slice of the
    # cell-level bind tuple.
    analysis = analyze_sql_cell(source, dialect=adapter.sqlglot_dialect)
    if analysis.parse_error:
        return _error_result(f"SQL parse error: {analysis.parse_error}", start_time)
    if not analysis.sql_body:
        return _error_result("Write SQL cell body is empty.", start_time)

    # Resolve cache policy — default to session when not specified.
    # Probe-based policies (fingerprint / snapshot) don't make
    # sense for writes; surface a diagnostic so the user knows.
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

    # Bind resolution + upstream-input hashes — same as read path.
    # Without these in provenance, an upstream variable change
    # wouldn't invalidate the write cell's cache and the seed would
    # silently use stale values.
    namespace, upstream_input_hashes = _load_upstream_variables(
        session, cell_id, analysis.references
    )
    try:
        params = resolve_bind_params(analysis.placeholder_positions, namespace)
    except BindError as exc:
        return _error_result(str(exc), start_time)

    # Provenance via the same hash function read cells use, with
    # the freshness/schema slots set to None (no probe). Run on
    # the runtime spec (paths rebased against the notebook dir)
    # so credential-file principal extraction works for relative
    # paths. ``read_only=False`` so the write-side principal
    # joins the cache identity for write cells.
    runtime_spec = _resolve_runtime_spec(spec, session.path)
    query_normalized = normalize_query(analysis.sql_body, adapter.sqlglot_dialect)
    connection_id = adapter.canonicalize_connection_id(runtime_spec, read_only=False)
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
        stats = _execute_write_statements(adapter, runtime_spec, analysis.sql_body, namespace)
    except Exception as exc:  # noqa: BLE001
        return _error_result(f"SQL execution failed: {_exception_message(exc)}", start_time)

    table = _synthesize_write_result_table(stats)
    blob = _serialize_arrow_ipc(table)
    # Use a higher row cap for write-cell status tables — the rows
    # are status entries (one per statement), not query results, so
    # truncating "5 of 6 statements" is unhelpful. Read cells keep
    # the default cap of 5.
    write_display_cap = max(20, table.num_rows)
    artifact = artifact_mgr.store_cell_output(
        cell_id=cell_id,
        variable_name=output_name,
        blob_data=blob,
        content_type="arrow/ipc",
        provenance_hash=var_provenance,
        source_hash=provenance_hash,
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


def _execute_write_statements(
    adapter: DriverAdapter,
    spec: ConnectionSpec,
    body: str,
    namespace: dict[str, Any],
) -> dict[str, Any]:
    """Open writable, split into statements, execute each with binds.

    Each split statement gets its own placeholder pass — the bind
    layer's allowlist still gates upstream values, and the
    statement is rewritten to the dialect's positional form before
    execute. This means ``INSERT INTO t VALUES (:n)`` works the
    same way in a write cell as it would in a read cell, just with
    the cell's mutating semantics.

    Returns ``{"statements": [{"kind": str, "rows_affected": int|None}, ...]}``.
    One entry per statement, in source order. ``rows_affected`` is
    None when the driver couldn't report a row count for that
    statement (DDL via PEP-249's ``cursor.rowcount = -1`` convention,
    or no rowcount reported at all). DML statements produce an
    integer; a 0 there is genuine ("UPDATE matched no rows"), not a
    sentinel.

    A failed statement aborts the run; partial state on disk is
    the user's problem (DDL/DML in SQLite isn't transactional by
    default, and Postgres' transaction semantics are driver-specific).

    Commit failures propagate. Earlier the implementation silenced
    all commit-time exceptions to paper over autocommit-mode
    "nothing to commit" warnings — that also hid real
    deferred-constraint, transaction, and transport failures.
    Surfacing the raw error means a failed commit produces a
    visible cell error instead of a misleading "success" with
    nothing actually persisted.
    """
    import sqlglot

    from strata.notebook.sql.analyzer import _extract_placeholder_positions

    parsed = [s for s in sqlglot.parse(body, dialect=adapter.sqlglot_dialect) if s]
    if not parsed:
        # sqlglot returned no statements — treat the whole body as a
        # single opaque statement (covers vendor-specific syntax we
        # can't fully parse). Placeholders still get extracted via
        # the regex path so :name bindings keep working.
        prepared = [(body, _statement_kind_from_text(body))]
    else:
        prepared = [
            (
                stmt.sql(dialect=adapter.sqlglot_dialect, comments=False),
                _statement_kind_from_expr(stmt),
            )
            for stmt in parsed
        ]

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
                # PEP 249 sentinel: -1 means "not available"; preserve
                # 0 as a real count (UPDATE matched zero rows). Only
                # ask for a count on DML — for DDL the concept doesn't
                # apply, and SQLite's ``changes()`` would inherit from
                # the prior DML statement, producing a misleading
                # number on a CREATE TABLE that happens to follow an
                # INSERT.
                rows_affected: int | None
                if _is_dml_kind(stmt_kind):
                    rc = getattr(cursor, "rowcount", -1)
                    if isinstance(rc, int) and rc >= 0:
                        rows_affected = rc
                    elif getattr(adapter, "name", None) == "sqlite":
                        # ADBC SQLite never populates cursor.rowcount;
                        # fall back to ``SELECT changes()`` which is
                        # SQLite's "rows modified by the last DML on
                        # this connection".
                        rows_affected = _sqlite_last_changes(conn)
                    else:
                        rows_affected = None
                else:
                    rows_affected = None
            finally:
                _safely_close(cursor)
            statements.append({"kind": stmt_kind, "rows_affected": rows_affected})

        # Commit explicitly. ADBC's DBAPI defaults autocommit=False
        # so user writes are buffered until commit. Surfaces any
        # commit-time error as the cell's failure — silencing was
        # the previous bug.
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
    """Run ``SELECT changes()`` to recover the last DML's row count.

    ADBC's SQLite driver doesn't populate ``cursor.rowcount`` (always
    returns -1). SQLite's own ``changes()`` returns the number of
    rows modified by the most recent INSERT / UPDATE / DELETE on the
    connection, which is exactly what we need. Errors surface as
    None so the display gracefully degrades to "—" instead of
    crashing the cell.
    """
    try:
        cur = conn.cursor()
        try:
            cur.execute("SELECT changes()")
            tbl = cur.fetch_arrow_table()
            rows = tbl.to_pylist()
            if rows:
                # ADBC returns the column under the literal expression
                # text "changes()"; iterate values defensively in case
                # a future ADBC release renames it.
                for v in rows[0].values():
                    if isinstance(v, int) and v >= 0:
                        return v
        finally:
            _safely_close(cur)
    except Exception:  # noqa: BLE001
        logger.exception("sqlite changes() probe failed")
    return None


def _statement_kind_from_expr(expr: Any) -> str:
    """Best-effort label for a parsed sqlglot statement.

    Returns strings like ``CREATE TABLE`` / ``DROP TABLE`` /
    ``INSERT`` / ``UPDATE`` / ``ALTER TABLE``. The label is what
    the synthesized result table surfaces in its ``kind`` column,
    so it should be self-explanatory at a glance.
    """
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
    """Fallback kind extractor when sqlglot can't parse the body.

    Strips leading comments / whitespace and returns the first
    keyword (uppercased). Better than ``UNKNOWN`` for vendor-
    specific or pragmatic SQL the parser doesn't fully understand.
    """
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
    """Per-statement Arrow table summarizing a write cell's execution.

    Schema: one row per statement, in source order, with
    ``stmt`` (1-indexed), ``kind`` (CREATE TABLE / INSERT / …),
    and ``rows_affected`` (nullable; None when the driver didn't
    report — typically DDL).
    """
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


def _resolve_runtime_spec(spec: ConnectionSpec, notebook_dir: Any) -> ConnectionSpec:
    """Return a spec copy with relative file paths resolved.

    The on-disk ``[connections.<name>]`` block is round-tripped
    verbatim — relative paths stay relative so a notebook can move
    between machines without notebook.toml needing edits. The
    runtime view (handed to the adapter) needs an absolute path
    because the server's process CWD is unrelated to the notebook
    directory.

    The same rule applies to driver-specific path-shaped fields
    that point at notebook-local files — currently
    ``credentials_path`` and ``write_credentials_path`` for the
    BigQuery driver. Without rebasing, a relative
    ``credentials_path = "creds/ro.json"`` resolves against the
    server's process CWD instead of the notebook directory and
    fails to open. ``uri`` round-trips as-is because SQLite's URI
    form already has well-defined absolute / relative semantics.
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

    # Driver-specific path fields. ``model_extra`` is where
    # extras live (BigQuery's credentials_path is an extra), so
    # the rebase has to update that dict too — Pydantic surfaces
    # extras both as attributes and through ``model_extra``, but
    # ``model_copy(update=...)`` only updates declared fields
    # by default. We pass them through anyway because Pydantic v2
    # also accepts unknown keys when ``extra='allow'`` is set on
    # the model (which ConnectionSpec uses).
    extras = getattr(spec, "model_extra", None) or {}
    for key in ("credentials_path", "write_credentials_path"):
        raw_value = extras.get(key) if key in extras else getattr(spec, key, None)
        new_value = _rebase(raw_value)
        if new_value != raw_value:
            update[key] = new_value

    if not update:
        return spec
    return spec.model_copy(update=update)


def _load_upstream_variables(
    session: NotebookSession,
    cell_id: str,
    references: list[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load upstream variable values + per-variable artifact hashes.

    Returns (namespace, upstream_input_hashes). The hashes feed into
    the provenance hash so a change in any referenced variable's
    artifact invalidates this cell's cache.
    """
    namespace: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    cell = next((c for c in session.notebook_state.cells if c.id == cell_id), None)
    if cell is None:
        return namespace, hashes

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
            blob = artifact_mgr.load_artifact_data(canonical_id, artifact.version)
            content_type = _content_type_of(artifact)
            namespace[var_name] = _deserialize_blob(blob, content_type)

    return namespace, hashes


def _content_type_of(artifact: Any) -> str:
    spec = getattr(artifact, "transform_spec", None)
    if not spec:
        return "json/object"
    try:
        parsed = json.loads(spec)
        return parsed.get("params", {}).get("content_type", "json/object")
    except (ValueError, KeyError):
        return "json/object"


def _deserialize_blob(blob: bytes, content_type: str) -> Any:
    """Pull just enough out of an artifact for SQL bind use.

    SQL binds want primitive values (int / str / bytes / etc.). The
    full ``serializer.deserialize`` round-trip handles every shape
    the notebook supports — for SQL we only need the scalar / Arrow
    paths, which simplifies the imports.
    """
    if content_type == "json/object":
        try:
            return json.loads(blob)
        except (ValueError, TypeError):
            return None
    if content_type == "arrow/ipc":
        # An upstream Arrow table can't be a SQL bind value; the
        # bind layer rejects it with a clear type error. We still
        # return a useful representation so the namespace is
        # populated and the user gets the right BindError.
        try:
            import pyarrow as pa

            reader = pa.ipc.open_stream(blob)
            return reader.read_all()
        except Exception:  # noqa: BLE001
            return blob
    if content_type == "pickle/object":
        import pickle

        try:
            return pickle.loads(blob)  # noqa: S301
        except Exception:  # noqa: BLE001
            return None
    return blob


def _run_probes(
    adapter: DriverAdapter,
    spec: ConnectionSpec,
    tables: list[Any],
    policy: Any,
) -> tuple[Any, Any]:
    """Run freshness + schema probes per the resolved policy.

    ``needs_separate_probe_conn=True`` (Postgres) gets its own
    connection because per-transaction stats are frozen inside the
    query connection's open transaction. Other adapters share the
    probe connection across both calls.
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


def _execute_query(
    adapter: DriverAdapter,
    spec: ConnectionSpec,
    analysis: Any,
    params: tuple[Any, ...],
) -> Any:
    """Open a read-only connection, run the rewritten query, fetch Arrow."""
    rewritten = rewrite_named_to_positional(analysis.sql_body, adapter.sqlglot_dialect)
    conn = adapter.open(spec, read_only=True)
    try:
        cursor = conn.cursor()
        try:
            if params:
                cursor.execute(rewritten, parameters=params)
            else:
                cursor.execute(rewritten)
            return cursor.fetch_arrow_table()
        finally:
            _safely_close(cursor)
    finally:
        _safely_close(conn)


def _exception_message(exc: BaseException) -> str:
    """Walk the exception chain so ADBC's wrapped errors stay visible.

    ADBC's Python driver often surfaces a generic ``InternalError``
    or ``OperationalError`` whose top-level message is
    ``"INTERNAL: (unknown error)"`` while the actually-useful
    "Failed to finalize statement: attempt to write a readonly
    database" lives in a chained exception. Without walking the
    chain, the user sees a useless message.
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


def _serialize_arrow_ipc(table: Any) -> bytes:
    import pyarrow as pa

    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


_WRITE_STATUS_COLUMNS = ("stmt", "kind", "rows_affected")


def _table_display(table: Any, *, max_rows: int = 5) -> dict[str, Any]:
    """Build a small markdown preview for the cell's display panel.

    ``max_rows`` caps the inline body. Read cells default to 5
    (query results can be huge, the user can ``LIMIT`` for more).
    Write-cell status tables pass a higher cap because the rows
    are status entries, not data — truncating "5 of 6 statements"
    is more confusing than helpful.

    Status tables are auto-detected by their canonical schema
    (``stmt, kind, rows_affected``) so a cache-hit display path —
    which doesn't know whether the cell was a write or read —
    still avoids truncation when re-rendering a cached write
    artifact.
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
    """Render a row-cell value for the markdown preview.

    ``None`` becomes an em-dash so a status table with nullable
    ``rows_affected`` reads as "no count reported" instead of the
    literal Python ``None`` repr.
    """
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

    # Cache hits update the cell's artifact map for the same
    # downstream-discovery reason as the miss path. Without this, a
    # cell-cache hit on a SQL cell after a notebook reopen would
    # leave artifact_uris empty and stale downstream caches.
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
