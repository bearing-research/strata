"""The access manifest — the experiment↔production data/model parity contract.

Scientists' first concern moving a notebook to AWS is that the pipeline reads
the *same* data and models as the experiment. This aggregates every external
access the compiled pipeline needs — Iceberg tables (with snapshot pins), data/
model mounts, SQL connections, and the secrets they require — so ops can grant
the Glue IAM role the same reach the notebook had, and a scientist can diff
notebook access against pipeline access.

It also flags **parity risks**: accesses that may differ between the notebook
run and the pipeline run (unpinned tables, read-write mounts, local files that
Glue cannot reach, external registries needing their own auth). The source-code
heuristics are best-effort — they surface likely risks, they do not prove
absence.
"""

from __future__ import annotations

import ast

from pydantic import BaseModel, Field

from strata.pipeline.ir import ConnectionRef, MountRef, PipelineIR, TableRef

# I/O calls whose first string-literal argument is a path we can check.
_IO_FUNCS = {
    "open",
    "read_csv",
    "read_parquet",
    "read_json",
    "read_excel",
    "read_feather",
    "read_table",
    "read_orc",
    "read_hdf",
    "read_pickle",
    "load",
    "load_model",
}
# Import roots that imply an external service the pipeline must authenticate to.
_EXTERNAL_ROOTS = {
    "mlflow": "MLflow",
    "sagemaker": "SageMaker",
}


class AccessManifest(BaseModel):
    """Everything the compiled pipeline must be able to reach, plus parity risks."""

    notebook_id: str
    notebook_name: str
    tables: list[TableRef] = Field(default_factory=list)
    mounts: list[MountRef] = Field(default_factory=list)
    connections: list[ConnectionRef] = Field(default_factory=list)
    required_env_vars: list[str] = Field(default_factory=list)
    parity_risks: list[str] = Field(default_factory=list)


def _source_risks(node_id: str, source: str) -> list[str]:
    """Best-effort scan for likely-undeclared external access in a cell body."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    risks: list[str] = []
    for stmt in ast.walk(tree):
        if isinstance(stmt, ast.Call):
            func = stmt.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in _IO_FUNCS and stmt.args:
                first = stmt.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    if "://" not in first.value:
                        risks.append(
                            f"node {node_id!r} reads a local path {first.value!r} "
                            "(not reachable from Glue; use an @mount)"
                        )
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                label = _EXTERNAL_ROOTS.get(alias.name.split(".")[0])
                if label:
                    risks.append(
                        f"node {node_id!r} imports {label}; production must "
                        "authenticate to the same model registry"
                    )
        elif isinstance(stmt, ast.ImportFrom) and stmt.module:
            label = _EXTERNAL_ROOTS.get(stmt.module.split(".")[0])
            if label:
                risks.append(
                    f"node {node_id!r} imports {label}; production must "
                    "authenticate to the same model registry"
                )
    return risks


def build_access_manifest(ir: PipelineIR) -> AccessManifest:
    """Aggregate the pipeline's external access surface + parity risks from *ir*."""
    mounts: dict[tuple[str, str, str], MountRef] = {}
    tables: dict[tuple[str, str], TableRef] = {}
    for node in ir.nodes:
        for m in node.mounts:
            mounts[(m.name, m.uri, m.mode)] = m
        for t in node.tables:
            tables[(t.name, t.uri)] = t

    risks: list[str] = []
    for t in tables.values():
        if t.snapshot_pin is None:
            risks.append(
                f"table {t.name!r} is not snapshot-pinned; the notebook and the "
                "pipeline can read different data if the table advances "
                "(pin a snapshot for exact parity)"
            )
    for m in mounts.values():
        if m.scheme == "file":
            risks.append(
                f"mount {m.name!r} is a local path ({m.uri!r}); Glue cannot reach "
                "it; move the data to s3:// for parity"
            )
        if m.mode == "rw":
            risks.append(
                f"mount {m.name!r} is read-write; its contents depend on run "
                "order and are not guaranteed identical between notebook and pipeline"
            )
    # SQL nodes run against their own engine in the pipeline (parity), but a
    # file-based connection (sqlite/duckdb on a local path) is not reachable
    # from Glue — the query would run against a different/empty database.
    seen_local_conn: set[str] = set()
    for node in ir.nodes:
        cfg = node.connection_config
        if not cfg or node.connection in seen_local_conn:
            continue
        path = str(cfg.get("path") or "")
        if cfg.get("driver") in ("sqlite", "duckdb") and path and "://" not in path:
            if path != ":memory:":
                seen_local_conn.add(node.connection or "")
                risks.append(
                    f"connection {node.connection!r} is a local {cfg['driver']} "
                    f"database ({path!r}); Glue cannot reach it; point it at a "
                    "networked warehouse for parity"
                )

    for node in ir.nodes:
        risks.extend(_source_risks(node.id, node.source))

    required_env: set[str] = set()
    for conn in ir.connections:
        required_env.update(conn.auth_env_vars)

    return AccessManifest(
        notebook_id=ir.notebook_id,
        notebook_name=ir.notebook_name,
        tables=list(tables.values()),
        mounts=list(mounts.values()),
        connections=ir.connections,
        required_env_vars=sorted(required_env),
        parity_risks=risks,
    )
