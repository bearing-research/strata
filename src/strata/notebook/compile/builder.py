"""Build a :class:`PipelineIR` from a notebook session or directory.

Read-only and offline: it reuses the session's already-built ``NotebookDag``
and per-cell analysis, adds a compile-time provenance preview, and maps each
translatable cell to a compute target. No cell is executed.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from strata.notebook.annotations import parse_annotations
from strata.notebook.dag import SweepProducer
from strata.notebook.env import compute_lockfile_hash
from strata.notebook.models import CellLanguage, CellState
from strata.notebook.provenance import (
    compute_provenance_hash,
    compute_source_hash,
    derive_subkey,
)
from strata.pipeline.ir import (
    ConnectionRef,
    MountRef,
    NodeInput,
    NodeOutput,
    NodeResources,
    PipelineEdge,
    PipelineIR,
    PipelineNode,
    PipelineParameter,
    TableRef,
    UnsupportedCell,
)

if TYPE_CHECKING:
    from strata.notebook.session import NotebookSession

# Languages that become pipeline nodes. The concrete compute target for a SQL
# cell is driver-dependent (see ``_sql_target``); Python is always Glue.
_NODE_LANGUAGES = {CellLanguage.PYTHON, CellLanguage.SQL}


class PipelineCompileError(Exception):
    """The notebook could not be compiled (no DAG — cycle or parse error)."""


def _cell_name(cell: CellState) -> str:
    """Resolved cell name from its ``@name`` annotation, or the cell id."""
    return parse_annotations(cell.source).name or cell.id


def _sql_connection(cell: CellState) -> str | None:
    """Connection name for a SQL cell (``@sql connection=…``), or None."""
    sql = parse_annotations(cell.source).sql
    return sql.connection if sql is not None else None


def _is_write_sql(cell: CellState) -> bool:
    """True for a ``@sql … write=true`` cell (mutates the warehouse)."""
    if cell.language is not CellLanguage.SQL:
        return False
    sql = parse_annotations(cell.source).sql
    return bool(sql is not None and sql.write)


def _sql_target(driver: str) -> str:
    """Compute target for a SQL cell: native Athena, else its own driver in Glue.

    Running a SQL cell on Athena is only faithful when the connection *is*
    Athena. For every other engine (postgres, snowflake, sqlite, …), parity
    means running the query against that same engine — which we do in a Glue
    Python Shell job via the driver (``glue_sql``).
    """
    return "athena" if driver == "athena" else "glue_sql"


_ENV_INDIRECTION = re.compile(r"\$\{(\w+)\}")


def _uri_scheme(uri: str) -> str:
    return uri.split("://", 1)[0] if "://" in uri else "file"


def _node_mounts(cell: CellState) -> list[MountRef]:
    """Data/model mounts a cell reads or writes (notebook-level + cell-level)."""
    return [
        MountRef(
            name=m.name,
            uri=m.uri,
            scheme=_uri_scheme(m.uri),
            mode=m.mode.value,
            option_keys=sorted(m.options.keys()),
        )
        for m in cell.mounts
    ]


def _node_tables(cell: CellState) -> list[TableRef]:
    """Iceberg tables a cell scans (``@table``); snapshot pin drives data parity."""
    return [
        TableRef(name=t.name, uri=t.uri, snapshot_pin=t.snapshot_pin)
        for t in parse_annotations(cell.source).tables
    ]


def _auth_env_vars(auth: dict[str, str]) -> list[str]:
    """The ``${VAR}`` indirection names a connection resolves at run time."""
    names: list[str] = []
    for value in auth.values():
        names.extend(_ENV_INDIRECTION.findall(str(value)))
    return sorted(set(names))


def _read_requirements(notebook_dir: Path) -> tuple[list[str], str]:
    """The notebook's declared runtime deps + requires-python (from pyproject)."""
    pyproject = notebook_dir / "pyproject.toml"
    if not pyproject.is_file():
        return [], ""
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    deps = [str(d) for d in project.get("dependencies", [])]
    return deps, str(project.get("requires-python", ""))


def build_pipeline_ir(
    session: NotebookSession, *, target: str = "aws", runtime: str = "container"
) -> PipelineIR:
    """Compile *session*'s notebook DAG into a :class:`PipelineIR`.

    Parameters
    ----------
    session
        A ``NotebookSession`` whose DAG is already analyzed.
    target
        Renderer family the IR is built for (only ``"aws"`` today).
    runtime
        Node execution model: ``"container"`` (Python 3.12 Lambda-container,
        the validated default with exact parity) or ``"glue"`` (Glue Python
        Shell / Athena). SQL nodes are supported in ``"glue"`` only for now.

    Raises
    ------
    PipelineCompileError
        If the notebook has no DAG (cycle or parse error).
    """
    dag = session.dag
    if dag is None:
        raise PipelineCompileError(
            "notebook DAG could not be built (cycle or parse error); "
            "fix the notebook before compiling"
        )
    state = session.notebook_state
    env_hash = compute_lockfile_hash(session.path)
    requirements, requires_python = _read_requirements(session.path)
    sql_ok = runtime == "glue"  # SQL nodes only compile under the Glue runtime today

    cells_by_id = {c.id: c for c in state.cells}
    connections_by_name = {c.name: c for c in state.connections}
    is_node = {
        cid: (
            c.language is CellLanguage.PYTHON
            or (c.language is CellLanguage.SQL and sql_ok and not _is_write_sql(c))
        )
        and cid not in dag.inactive_cells
        for cid, c in cells_by_id.items()
    }

    parameters: list[PipelineParameter] = []
    unsupported: list[UnsupportedCell] = []
    warnings: list[str] = []
    seen_params: set[str] = set()

    # Widget cells contribute their defined controls as pipeline parameters.
    for cell in state.cells:
        if cell.language is CellLanguage.WIDGET and cell.id not in dag.inactive_cells:
            for var in cell.defines:
                if var not in seen_params:
                    parameters.append(PipelineParameter(name=var, from_cell=cell.id))
                    seen_params.add(var)

    # Prompt / R cells have no v1 compute target — record, do not drop.
    for cell in state.cells:
        if cell.id in dag.inactive_cells:
            continue
        if cell.language is CellLanguage.PROMPT:
            unsupported.append(
                UnsupportedCell(
                    id=cell.id,
                    language="prompt",
                    reason="prompt cells are deferred past compile v1 (needs an LLM-call target)",
                )
            )
        elif cell.language is CellLanguage.R:
            unsupported.append(
                UnsupportedCell(
                    id=cell.id,
                    language="r",
                    reason="R cells have no AWS compute target in compile v1",
                )
            )
        elif _is_write_sql(cell):
            unsupported.append(
                UnsupportedCell(
                    id=cell.id,
                    language="sql",
                    reason="write SQL cells (write=true) mutate the warehouse; "
                    "not executed by compile v1",
                )
            )
        elif cell.language is CellLanguage.SQL and not sql_ok:
            unsupported.append(
                UnsupportedCell(
                    id=cell.id,
                    language="sql",
                    reason="SQL nodes are not yet supported in the container runtime; "
                    "use --runtime glue for SQL",
                )
            )

    # Per-(producer, variable) output identity, filled in topological order so a
    # consumer always sees its upstream outputs already computed.
    output_static_id: dict[tuple[str, str], str] = {}

    nodes: list[PipelineNode] = []
    edges: list[PipelineEdge] = []

    for cid in dag.topological_order:
        cell = cells_by_id.get(cid)
        if cell is None or not is_node.get(cid, False):
            continue

        inputs: list[NodeInput] = []
        input_hashes: list[str] = []
        seen_vars: set[str] = set()
        for var in cell.references:
            if var in seen_vars:
                continue
            producer = dag.variable_producer.get(var)
            if producer is None:
                continue  # free variable / builtin / undefined upstream
            seen_vars.add(var)
            if isinstance(producer, SweepProducer):
                warnings.append(
                    f"node {cid!r} consumes {var!r} from a sweep producer; "
                    "sweep fan-out is not supported in compile v1"
                )
                continue
            if is_node.get(producer, False):
                path = f"{producer}/{var}.artifact"
                inputs.append(NodeInput(variable=var, from_node=producer, artifact_path=path))
                edges.append(PipelineEdge(from_node=producer, to_node=cid, variable=var))
                oid = output_static_id.get((producer, var))
                if oid is not None:
                    input_hashes.append(oid)
            elif producer in cells_by_id and cells_by_id[producer].language is CellLanguage.WIDGET:
                inputs.append(NodeInput(variable=var, from_param=var))
            else:
                prod_lang = cells_by_id[producer].language.value if producer in cells_by_id else "?"
                warnings.append(
                    f"node {cid!r} consumes {var!r} from unsupported {prod_lang} cell {producer!r}"
                )

        source_hash = compute_source_hash(cell.source)
        node_prov = compute_provenance_hash(input_hashes, source_hash, env_hash)

        outputs: list[NodeOutput] = []
        for var in sorted(dag.consumed_variables.get(cid, set())):
            outputs.append(NodeOutput(variable=var, artifact_path=f"{cid}/{var}.artifact"))
            output_static_id[(cid, var)] = derive_subkey(node_prov, var)

        connection: str | None = None
        connection_config: dict | None = None
        if cell.language is CellLanguage.SQL:
            connection = _sql_connection(cell)
            spec = connections_by_name.get(connection) if connection else None
            driver = spec.driver if spec is not None else "unknown"
            compute_target = _sql_target(driver)
            connection_config = spec.model_dump(mode="json") if spec is not None else None
        else:
            compute_target = "container" if runtime == "container" else "glue_python_shell"

        nodes.append(
            PipelineNode(
                id=cid,
                name=_cell_name(cell),
                kind=cell.language.value,
                compute_target=compute_target,
                source=cell.source,
                inputs=inputs,
                outputs=outputs,
                static_provenance=node_prov,
                env=dict(cell.env or {}),
                connection=connection,
                connection_config=connection_config,
                mounts=_node_mounts(cell),
                tables=_node_tables(cell),
                resources=NodeResources(timeout=cell.timeout, worker=cell.worker),
            )
        )

    # `# @after` ordering-only edges (no variable flows, DagEdge.variable == "").
    # They carry no artifact so they are not inputs, but execution order must
    # still respect them — e.g. a SQL cell that reads a table a prior cell writes.
    for edge in dag.edges:
        if (
            edge.variable == ""
            and is_node.get(edge.from_cell_id, False)
            and is_node.get(edge.to_cell_id, False)
        ):
            edges.append(
                PipelineEdge(from_node=edge.from_cell_id, to_node=edge.to_cell_id, variable="")
            )

    topo_nodes = [cid for cid in dag.topological_order if is_node.get(cid, False)]

    # Connections the pipeline actually uses (referenced by a SQL node).
    used_connections = {n.connection for n in nodes if n.connection}
    connections = [
        ConnectionRef(
            name=conn.name,
            driver=conn.driver,
            auth_env_vars=_auth_env_vars(conn.auth),
        )
        for conn in state.connections
        if conn.name in used_connections
    ]

    return PipelineIR(
        target=target,
        runtime=runtime,
        pipeline_id=state.id,
        pipeline_name=state.name,
        env_hash=env_hash,
        requirements=requirements,
        requires_python=requires_python,
        nodes=nodes,
        edges=edges,
        topological_order=topo_nodes,
        parameters=parameters,
        connections=connections,
        unsupported=unsupported,
        warnings=warnings,
    )


def build_pipeline_ir_from_dir(
    notebook_dir: Path, *, target: str = "aws", runtime: str = "container"
) -> PipelineIR:
    """Load the notebook at *notebook_dir* and compile it (offline, read-only)."""
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession

    notebook_dir = Path(notebook_dir)
    state = parse_notebook(notebook_dir)
    session = NotebookSession(state, notebook_dir)
    return build_pipeline_ir(session, target=target, runtime=runtime)
