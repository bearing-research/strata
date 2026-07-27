"""Author a pipeline as plain ``@node``-decorated Python functions.

A second frontend over the same engine the notebook uses. It lowers a set of
``@node`` functions into the identical :class:`~strata.pipeline.PipelineIR`
that :mod:`strata.pipeline` renders and runs — proving the engine is
frontend-agnostic (the notebook is just one producer of the IR).

The mapping mirrors the notebook's, but reads the graph off function signatures
instead of AST variable analysis:

* a function's **parameters** are its inputs, resolved to an upstream node's
  output of the same name (each output name is defined by exactly one node);
* the **dict it returns** declares its outputs (``return {"model": clf}``);
* the function **body** (minus the return, with ``name = expr`` appended per
  returned key) is the node source the runtime execs — inputs arrive as locals
  and each declared output is read back by name, exactly like a notebook cell.

Functions are listed in dependency order (an input's producer must appear
earlier); the engine handles everything below the IR.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from typing import Any, NamedTuple

from strata.notebook.provenance import (
    compute_provenance_hash,
    compute_source_hash,
    derive_subkey,
)
from strata.pipeline.ir import (
    NodeInput,
    NodeOutput,
    NodeResources,
    PipelineEdge,
    PipelineIR,
    PipelineNode,
)

_META_ATTR = "__strata_node__"


class _Lowered(NamedTuple):
    """A ``@node`` function reduced to its IR-relevant parts."""

    id: str
    name: str
    timeout: int | None
    worker: str | None
    inputs: list[str]
    outputs: list[str]
    source: str


class PipelineCompileError(Exception):
    """A set of ``@node`` functions could not be lowered into a pipeline IR."""


def _node_id(fn: Callable) -> str:
    """A node's id: the function's ``__name__``."""
    return getattr(fn, "__name__", repr(fn))


def node(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    timeout: int | None = None,
    worker: str | None = None,
) -> Callable:
    """Mark a function as a pipeline node.

    Usable bare (``@node``) or parameterized (``@node(name=..., timeout=...)``).
    Attaches metadata; the function stays callable unchanged (so the same module
    runs locally and compiles to a pipeline).
    """

    def wrap(f: Callable) -> Callable:
        setattr(
            f,
            _META_ATTR,
            {"name": name or _node_id(f), "timeout": timeout, "worker": worker},
        )
        return f

    return wrap(fn) if fn is not None else wrap


def _lower(fn: Callable) -> tuple[list[str], list[str], str]:
    """Return ``(input_names, output_names, node_source)`` for a ``@node`` fn."""
    fn_id = _node_id(fn)
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    func = tree.body[0]
    if not isinstance(func, ast.FunctionDef):
        raise PipelineCompileError(f"@node {fn_id!r} must be a plain (non-async) function")

    inputs = [arg.arg for arg in func.args.args]
    body = list(func.body)
    ret = body.pop() if body else None
    if not isinstance(ret, ast.Return) or not isinstance(ret.value, ast.Dict):
        raise PipelineCompileError(
            f"@node {fn_id!r} must end with `return {{...}}` declaring its outputs"
        )

    lines = [ast.unparse(stmt) for stmt in body]
    outputs: list[str] = []
    for key, value in zip(ret.value.keys, ret.value.values):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise PipelineCompileError(f"@node {fn_id!r} output keys must be string literals")
        outputs.append(key.value)
        # `return {"x": x}` already binds x in the body; only bind when the
        # returned value is some other expression (`return {"x": compute()}`).
        if not (isinstance(value, ast.Name) and value.id == key.value):
            lines.append(f"{key.value} = {ast.unparse(value)}")
    return inputs, outputs, "\n".join(lines) + "\n"


def build_pipeline_ir_from_functions(
    functions: list[Callable],
    *,
    name: str,
    target: str = "aws",
    runtime: str = "container",
    env_hash: str = "",
    requirements: list[str] | None = None,
    requires_python: str = "",
) -> PipelineIR:
    """Compile ``@node`` *functions* (in dependency order) into a pipeline IR.

    Raises
    ------
    PipelineCompileError
        On a non-``@node`` function, a missing ``return {...}``, a duplicated
        output name, or an input with no upstream (or forward-referenced)
        producer.
    """
    if runtime != "container":
        raise PipelineCompileError(
            f"the @node frontend targets the container runtime; got {runtime!r}"
        )

    lowered: list[_Lowered] = []
    for fn in functions:
        meta = getattr(fn, _META_ATTR, None)
        if meta is None:
            raise PipelineCompileError(f"function {_node_id(fn)!r} is not decorated with @node")
        inputs, outputs, source = _lower(fn)
        lowered.append(
            _Lowered(
                _node_id(fn), meta["name"], meta["timeout"], meta["worker"], inputs, outputs, source
            )
        )

    # Each output name is defined by exactly one node (no redefinition).
    producer_of: dict[str, str] = {}
    for lo in lowered:
        for var in lo.outputs:
            if var in producer_of:
                raise PipelineCompileError(
                    f"output {var!r} is defined by both {producer_of[var]!r} and {lo.id!r}; "
                    "output names must be unique across nodes"
                )
            producer_of[var] = lo.id

    # Which of a node's outputs a downstream node consumes (only those persist).
    consumed: dict[str, set[str]] = {}
    for lo in lowered:
        for var in lo.inputs:
            producer = producer_of.get(var)
            if producer is not None:
                consumed.setdefault(producer, set()).add(var)

    output_static_id: dict[tuple[str, str], str] = {}
    seen_ids: set[str] = set()
    nodes: list[PipelineNode] = []
    edges: list[PipelineEdge] = []

    for lo in lowered:
        node_id, inputs, outputs, source = lo.id, lo.inputs, lo.outputs, lo.source
        node_inputs: list[NodeInput] = []
        input_hashes: list[str] = []
        for var in inputs:
            producer = producer_of.get(var)
            if producer is None:
                raise PipelineCompileError(
                    f"node {node_id!r} input {var!r} has no upstream producer"
                )
            if producer not in seen_ids:
                raise PipelineCompileError(
                    f"node {node_id!r} references {var!r} from {producer!r}, which is defined "
                    "later; list functions in dependency order"
                )
            path = f"{producer}/{var}.artifact"
            node_inputs.append(NodeInput(variable=var, from_node=producer, artifact_path=path))
            edges.append(PipelineEdge(from_node=producer, to_node=node_id, variable=var))
            input_hashes.append(output_static_id[(producer, var)])

        source_hash = compute_source_hash(source)
        node_prov = compute_provenance_hash(input_hashes, source_hash, env_hash)

        node_outputs: list[NodeOutput] = []
        for var in sorted(consumed.get(node_id, set())):
            node_outputs.append(NodeOutput(variable=var, artifact_path=f"{node_id}/{var}.artifact"))
            output_static_id[(node_id, var)] = derive_subkey(node_prov, var)

        nodes.append(
            PipelineNode(
                id=node_id,
                name=lo.name,
                kind="python",
                compute_target="container",
                source=source,
                inputs=node_inputs,
                outputs=node_outputs,
                static_provenance=node_prov,
                resources=NodeResources(timeout=lo.timeout, worker=lo.worker),
            )
        )
        seen_ids.add(node_id)

    return PipelineIR(
        target=target,
        runtime=runtime,
        pipeline_id=name.lower().replace(" ", "-"),
        pipeline_name=name,
        env_hash=env_hash,
        requirements=requirements or [],
        requires_python=requires_python,
        nodes=nodes,
        edges=edges,
        topological_order=[n.id for n in nodes],
    )


def build_pipeline_ir_from_module(
    module: Any, *, name: str | None = None, **kwargs: Any
) -> PipelineIR:
    """Collect a *module*'s ``@node`` functions (definition order) and compile them.

    A module's ``__dict__`` preserves definition order, which is the order the
    functions must already be in (an input's producer comes first).
    """
    functions: list[Callable] = [
        obj for obj in vars(module).values() if callable(obj) and hasattr(obj, _META_ATTR)
    ]
    return build_pipeline_ir_from_functions(
        functions, name=name or getattr(module, "__name__", "pipeline"), **kwargs
    )
