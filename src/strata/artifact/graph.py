"""Strata's nodes in the ledger's structural graph (ADR 002 §6.3).

A cell definition is one node under ``strata/cell``: its payload is the executor and
the params that define the cell (source, environment, pins), never what a run recorded
about itself (its duration, machine, download times) or which output of the cell a
version is. Its edges are the inputs the run read: one per upstream variable, under
the variable's name, pointing at the node that produced the upstream version; one per
fetched URL, dataset and table read, as leaf nodes. Derivation is one-way: nothing reads the graph
to decide what to run.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping

from vestigo.graph import Edge, NodeId, NodeSpec
from vestigo.provenance import TransformSpec
from vestigo.store import ArtifactStore

logger = logging.getLogger(__name__)

CELL_DOMAIN = "strata/cell"
RECONSTRUCTED_DOMAIN = "strata/cell-reconstructed"
CELL_EXECUTOR = "notebook/cell@v1"
CELL_DEFINITIONS_UPGRADE = "cell_definitions_reconstructed"

# What a run records about itself: two runs of one definition differ here.
_RECORDED_ONLY = frozenset(
    {"source", "build_env", "build_duration_ms", "hardware", "fetched_at", "loop_run"}
)
# Which output of the definition a version is: the definition is the same for all of them.
_OUTPUT_ONLY = frozenset({"variable_name", "content_type", "display", "display_count"})

# The inverse of ``NotebookArtifactManager.cell_artifact_id``: a cell id is a UUID prefix,
# so the first ``_var_`` after it starts the variable, which an ``@iter``/``@variant`` may end.
_CELL_OUTPUT_ID = re.compile(r"^nb_.+?_cell_[0-9a-f]+_var_(?P<variable>[^@]+)")
_ARTIFACT_URI = "strata://artifact/"


def definition_payload(spec: TransformSpec) -> bytes:
    """The bytes that identify a cell definition: the executor and its defining params."""
    params = {k: v for k, v in spec.params.items() if k not in _RECORDED_ONLY | _OUTPUT_ONLY}
    return json.dumps(
        {"executor": spec.executor, "params": params}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def is_cell_output(spec: TransformSpec | None) -> bool:
    """Whether a stored spec is one ``stage_cell_output`` wrote: a cell's variable."""
    return spec is not None and spec.executor == CELL_EXECUTOR and "variable_name" in spec.params


def _leaf(uri: str, location: str | None = None) -> NodeSpec:
    """An external input as a node: its URI without the version, under its kind's domain.

    A table read that resolved carries the table's location on a second line: a bare
    ``ns.table`` names a different table under each default catalog, and the URI
    alone would make them one leaf. A pinned read is not loaded, so it has none.
    """
    bare = uri.partition("@v=")[0]
    if uri.startswith("strata://name/"):
        domain = "strata/dataset"
    elif uri.startswith(("http://", "https://")):
        domain = "strata/fetch"
    elif uri.startswith(_ARTIFACT_URI):
        domain = "strata/artifact"
    else:
        domain = "strata/table"
        if location:
            bare = f"{bare}\n{location}"
    return NodeSpec(domain=domain, payload=bare.encode("utf-8"))


def _role_of(uri: str) -> str:
    """The port an input binds to: the upstream variable's name, or its kind."""
    if uri.startswith(_ARTIFACT_URI):
        match = _CELL_OUTPUT_ID.match(uri.removeprefix(_ARTIFACT_URI))
        return match.group("variable") if match else "input"
    if uri.startswith("strata://name/"):
        return "dataset"
    if uri.startswith(("http://", "https://")):
        return "fetch"
    return "table"


def _target(
    store: ArtifactStore,
    uri: str,
    known: Mapping[tuple[str, int], NodeId],
    tenant: str | None,
) -> NodeId | None:
    """The node that produced an upstream version, when one was recorded in this scope."""
    if not uri.startswith(_ARTIFACT_URI):
        return None
    artifact_id, sep, version = uri.removeprefix(_ARTIFACT_URI).partition("@v=")
    if not sep or not version.isdigit():
        return None
    key = (artifact_id, int(version))
    return known.get(key) or store.definition_of(*key, tenant=tenant)


def _definition(
    store: ArtifactStore,
    domain: str,
    payload: bytes,
    input_uris: list[str],
    *,
    positional: bool,
    tenant: str | None = None,
    known: Mapping[tuple[str, int], NodeId] | None = None,
    locations: Mapping[str, str] | None = None,
) -> list[NodeSpec]:
    """Leaves first, then the cell's node, as one batch for ``put_batch``.

    *locations* is each resolved table's location by its URI.
    """
    specs: list[NodeSpec] = []
    edges: list[Edge] = []
    positions: dict[str, int] = {}
    for index, uri in enumerate(sorted(input_uris)):
        if positional:
            role, position = "input", index
        else:
            role = _role_of(uri)
            position = positions.get(role, 0)
            positions[role] = position + 1
        target: NodeId | int | None = _target(store, uri, known or {}, tenant)
        if target is None:
            specs.append(_leaf(uri, (locations or {}).get(uri)))
            target = len(specs) - 1
        edges.append(Edge(role=role, position=position, target=target))
    specs.append(NodeSpec(domain=domain, payload=payload, edges=tuple(edges)))
    return specs


def cell_definition(
    store: ArtifactStore,
    spec: TransformSpec,
    input_versions: Mapping[str, str] | None,
    *,
    locations: Mapping[str, str] | None = None,
    tenant: str | None = None,
) -> list[NodeSpec]:
    """The structure a cell run publishes with its outputs (domain ``strata/cell``).

    Every key of ``input_versions`` is an input: an upstream variable's version, a
    fetch, a dataset, or a table read (its URI, the snapshot as the value), which
    becomes a ``strata/table`` leaf. *locations*, by table URI, is where each read
    that resolved found its table; the leaf carries it.
    """
    return _definition(
        store,
        CELL_DOMAIN,
        definition_payload(spec),
        list(input_versions or ()),
        positional=False,
        tenant=tenant,
        locations=locations,
    )


def reconstruct_definitions(store: ArtifactStore) -> None:
    """Give the versions stored before the graph a node each, once.

    Their roles were never recorded, so the node is built from the stored spec with
    role ``input`` in positional order under ``strata/cell-reconstructed``: partially
    reconstructed provenance, never replaced. Oldest first, so an upstream's node
    exists when its reader's edge is written.
    """
    if store.upgrade_applied(CELL_DEFINITIONS_UPGRADE):
        return
    filled: dict[tuple[str, int], NodeId] = {}
    by_tenant: dict[str | None, dict[tuple[str, int], NodeId]] = {}
    for version in store.versions_lacking_definition(executor=CELL_EXECUTOR):
        spec = version.spec
        if not is_cell_output(spec):
            continue
        assert spec is not None
        inputs = json.loads(version.input_versions) if version.input_versions else {}
        specs = _definition(
            store,
            RECONSTRUCTED_DOMAIN,
            definition_payload(spec),
            list(inputs),
            positional=True,
            tenant=version.tenant or None,
            known=filled,
        )
        node = store.graph.put_batch(specs, tenant=version.tenant or None)[-1]
        filled[(version.id, version.version)] = node
        by_tenant.setdefault(version.tenant or None, {})[(version.id, version.version)] = node
    # A definition is checked against its tenant's scope, so one call per tenant.
    for tenant, definitions in by_tenant.items():
        store.set_definitions(definitions, tenant=tenant)
    store.mark_upgrade_applied(CELL_DEFINITIONS_UPGRADE)
    if filled:
        logger.info(
            "Reconstructed the definitions of %d versions stored before the graph", len(filled)
        )
