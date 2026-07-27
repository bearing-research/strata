"""Runtime for a compiled Python pipeline node (the Glue Python Shell body).

A generated Glue job is a third cell-execution path alongside ``harness.py``
and ``pool_worker.py``; to stay faithful to the notebook it reuses the very
same ``serializer.serialize_value`` / ``deserialize_value`` — so a value
round-trips through S3 byte-for-byte the way it would between notebook cells.

Data crosses node boundaries **by reference**: each consumed variable is one
object in the store (any URI fsspec understands — ``s3://``, ``gs://``,
``az://``, or a local path) with a sibling ``<uri>.meta.json`` recording the
serializer ``content_type``. A node reads its inputs' blobs, execs the cell
source, and writes its outputs plus their sidecars.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import fsspec

from strata.notebook.provenance import (
    compute_provenance_hash,
    compute_source_hash,
    derive_subkey,
)
from strata.notebook.serializer import deserialize_value, serialize_value

_META_SUFFIX = ".meta.json"
_PROV_SUFFIX = ".prov"


def _read_bytes(uri: str) -> bytes:
    with fsspec.open(uri, "rb") as fh:
        return fh.read()


def _write_bytes(uri: str, data: bytes) -> None:
    with fsspec.open(uri, "wb") as fh:
        fh.write(data)


def _read_input(var: str, uri: str, workdir: Path):
    """Fetch an input artifact + its sidecar and deserialize it to a value."""
    meta = json.loads(_read_bytes(uri + _META_SUFFIX))
    content_type = meta["content_type"]
    local = workdir / f"__in_{var}"
    _write_bytes(str(local), _read_bytes(uri))
    return deserialize_value(content_type, local)


def _write_output(var: str, uri: str, value, provenance: str, workdir: Path) -> str:
    """Serialize *value* to *uri*, with a content-type sidecar + provenance marker."""
    payload = serialize_value(value, workdir, var)
    blob = (workdir / payload["file"]).read_bytes()
    _write_bytes(uri, blob)
    _write_bytes(
        uri + _META_SUFFIX,
        json.dumps({"content_type": payload["content_type"], "variable": var}).encode("utf-8"),
    )
    _write_bytes(uri + _PROV_SUFFIX, derive_subkey(provenance, var).encode("utf-8"))
    return uri


def _read_prov_marker(uri: str) -> str | None:
    """Read the ``<uri>.prov`` provenance marker, or None if absent."""
    try:
        return _read_bytes(uri + _PROV_SUFFIX).decode("utf-8")
    except (FileNotFoundError, OSError):
        return None


def _input_hash(uri: str) -> str:
    """Provenance hash of an input artifact: its producer's per-variable marker.

    Falls back to hashing the blob bytes when no marker is present (e.g. an
    externally-staged input), so a change to the data still invalidates.
    """
    marker = _read_prov_marker(uri)
    if marker is not None:
        return marker
    return hashlib.sha256(_read_bytes(uri)).hexdigest()


def _node_provenance(source: str, inputs: dict[str, str], env_hash: str) -> str:
    """Provenance of this node: input markers + source + env, like the notebook."""
    input_hashes = [_input_hash(uri) for uri in inputs.values()]
    return compute_provenance_hash(input_hashes, compute_source_hash(source), env_hash)


def _outputs_fresh(provenance: str, outputs: dict[str, str]) -> bool:
    """True when every output already on the store matches *provenance*.

    Markers are co-located with the outputs, so a match means the exact result
    of this computation is already present — safe to skip. A changed source or
    input yields a different provenance, so stale outputs never match.
    """
    if not outputs:
        return False
    for var, uri in outputs.items():
        if _read_prov_marker(uri) != derive_subkey(provenance, var):
            return False
    return True


def run_python_node(
    *,
    source: str,
    inputs: dict[str, str],
    outputs: dict[str, str],
    env: dict[str, str] | None = None,
    env_hash: str = "",
    skip_if_fresh: bool = True,
    workdir: str | Path | None = None,
) -> dict:
    """Execute one Python node: read inputs, exec *source*, write outputs.

    Parameters
    ----------
    source
        The cell body (annotations are inert comments, kept as-is so execution
        matches the notebook harness exactly).
    inputs, outputs
        ``{variable: uri}`` maps. Inputs are deserialized and bound in the exec
        namespace; each output variable is read back from the namespace after
        exec and serialized to its uri.
    env
        Environment variables to set for the duration of the exec (restored
        afterwards).
    env_hash
        The notebook's environment hash, folded into provenance so a dependency
        change invalidates the skip.
    skip_if_fresh
        When True (default), skip compute if every output already on the store
        matches this node's provenance (provenance-aware handoff).
    workdir
        Scratch directory for (de)serialization; a temp dir by default.

    Returns
    -------
    dict
        ``{"outputs": {var: uri}, "provenance": str, "skipped": bool}``.
    """
    work = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="strata-node-"))
    work.mkdir(parents=True, exist_ok=True)

    provenance = _node_provenance(source, inputs, env_hash)
    if skip_if_fresh and _outputs_fresh(provenance, outputs):
        return {"outputs": dict(outputs), "provenance": provenance, "skipped": True}

    namespace: dict[str, object] = {}
    for var, uri in inputs.items():
        namespace[var] = _read_input(var, uri, work)

    saved: dict[str, str | None] = {}
    if env:
        for key, value in env.items():
            saved[key] = os.environ.get(key)
            os.environ[key] = value
    try:
        exec(compile(source, "<strata-node>", "exec"), namespace)  # noqa: S102 — running the cell
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old

    written: dict[str, str] = {}
    for var, uri in outputs.items():
        if var not in namespace:
            raise KeyError(f"node did not define output variable {var!r}")
        written[var] = _write_output(var, uri, namespace[var], provenance, work)
    return {"outputs": written, "provenance": provenance, "skipped": False}


def _resolve_indirections(auth: dict[str, str]) -> dict[str, str]:
    """Substitute ``${VAR}`` in connection auth values from the environment.

    Matches how the notebook resolves connection secrets at open time, so the
    pipeline authenticates to the same engine with the same credentials.
    """
    import re

    def sub(value: str) -> str:
        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), value)

    return {key: sub(str(val)) for key, val in auth.items()}


def run_sql_node(
    *,
    source: str,
    connection_config: dict,
    inputs: dict[str, str],
    outputs: dict[str, str],
    env: dict[str, str] | None = None,
    env_hash: str = "",
    skip_if_fresh: bool = True,
    workdir: str | Path | None = None,
) -> dict:
    """Execute one SQL node against its own connection engine (parity path).

    Reuses the notebook's read path — ``get_adapter`` → ``analyze_sql_cell`` →
    ``resolve_bind_params`` → ``rewrite_named_to_positional`` → open the driver
    read-only → ``fetch_arrow_table`` — so the pipeline runs the identical query
    on the identical engine the experiment used. The Arrow result is serialized
    to each output uri like any other node output. Provenance-aware skip works
    the same way as :func:`run_python_node`.

    Returns ``{"outputs": {var: uri}, "provenance": str, "skipped": bool}``.
    """
    from strata.notebook.models import ConnectionSpec
    from strata.notebook.sql.analyzer import analyze_sql_cell, rewrite_named_to_positional
    from strata.notebook.sql.bind import resolve_bind_params
    from strata.notebook.sql.registry import get_adapter

    work = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="strata-sql-"))
    work.mkdir(parents=True, exist_ok=True)

    provenance = _node_provenance(source, inputs, env_hash)
    if skip_if_fresh and _outputs_fresh(provenance, outputs):
        return {"outputs": dict(outputs), "provenance": provenance, "skipped": True}

    saved: dict[str, str | None] = {}
    if env:
        for key, value in env.items():
            saved[key] = os.environ.get(key)
            os.environ[key] = value
    try:
        spec = ConnectionSpec(**connection_config)
        if spec.auth:
            spec = spec.model_copy(update={"auth": _resolve_indirections(spec.auth)})
        adapter = get_adapter(spec.driver)
        analysis = analyze_sql_cell(source, dialect=adapter.sqlglot_dialect)

        namespace = {var: _read_input(var, uri, work) for var, uri in inputs.items()}
        params = resolve_bind_params(analysis.placeholder_positions, namespace)
        rewritten = rewrite_named_to_positional(analysis.sql_body, adapter.sqlglot_dialect)

        conn = adapter.open(spec, read_only=True)
        try:
            cursor = conn.cursor()
            try:
                if params:
                    cursor.execute(rewritten, parameters=params)
                else:
                    cursor.execute(rewritten)
                table = cursor.fetch_arrow_table()
            finally:
                cursor.close()
        finally:
            conn.close()
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old

    written = {
        var: _write_output(var, uri, table, provenance, work) for var, uri in outputs.items()
    }
    return {"outputs": written, "provenance": provenance, "skipped": False}
