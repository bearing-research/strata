"""A cell's execution key, from cellgraph.

cellgraph assembles what a cell reads upstream and keys Python and R cells. Strata
says what each upstream published (its artifacts) and what each input resolved to
(the fingerprints staleness and the executor already make). A prompt or SQL cell is
host-keyed: cellgraph gives its upstream parts, this module adds its inputs, and the
key is ``parts_digest`` over both, as for the others.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cellgraph as cg

from strata.notebook.provenance import cell_provenance_inputs, compute_provenance_hash

if TYPE_CHECKING:
    from strata.notebook.executor_host import ExecutorHost


class InputNotFingerprinted(RuntimeError):
    """An input the cell declares is missing from the fingerprints it was keyed over."""


@dataclass(frozen=True)
class CellKey:
    """A run's provenance hash, and the upstream components it covers."""

    provenance_hash: str
    upstream: list[str]
    # What it was built from, to explain a later change. None when nothing is wired.
    parts: dict[str, Any] | None = None


class _Lookups(Mapping[Any, Any]):
    """Keys known up front, each value read from the store only when asked for."""

    def __init__(self, keys: list[Any], read: Callable[[Any], Any]) -> None:
        self._keys = keys
        self._read = read

    def __getitem__(self, key: Any) -> Any:
        if key not in self._keys:
            raise KeyError(key)
        value = self._read(key)
        if value is None:
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[Any]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


def published(session: ExecutorHost, analysis: cg.Analysis) -> Callable[[str], cg.Published]:
    """What each cell published, as cellgraph reads it: its artifacts in the store."""
    manager = session.get_artifact_manager()
    store = manager.artifact_store

    def ref(artifact: Any, by_content: bool) -> cg.OutputRef:
        digest = None
        if by_content:
            # Recorded at finalize; backfilled once if absent.
            digest = artifact.content_sha256 or store.content_digest(artifact.id, artifact.version)
        return cg.OutputRef(cg.Provenance(artifact.provenance_hash), digest)

    def from_uri(uri: str, by_content: bool) -> cg.OutputRef | cg.Unavailable:
        try:
            artifact_id, version = session.parse_artifact_uri(uri)
        except ValueError:
            return cg.Unavailable(f"not an artifact URI: {uri}")
        artifact = store.get_artifact(artifact_id, version)
        if artifact is None:
            return cg.Unavailable(f"{uri} is not in the store")
        return ref(artifact, by_content)

    def of(cell_id: str) -> cg.Published:
        cell = session.notebook_state.get_cell(cell_id)
        if cell is None:
            return cg.Published()
        upstream = analysis.cells.get(cell_id)
        # A value that outlives its key is read by its bytes.
        by_content = upstream is not None and upstream.nocache
        uris = dict(cell.artifact_uris)
        fanout = analysis.graph.fanout_cells.get(cell_id)
        names = [name for name, _ in fanout.variants] if fanout is not None else []

        def instance(key: tuple[str, str]) -> cg.OutputRef | None:
            var, name = key
            found = store.get_latest_version(manager.cell_artifact_id(cell_id, var, variant=name))
            return ref(found, by_content) if found is not None else None

        cell_ref = None
        if not uris and cell.artifact_uri:
            cell_ref = from_uri(cell.artifact_uri, by_content)
        return cg.Published(
            outputs=_Lookups(list(uris), lambda var: from_uri(uris[var], by_content)),
            instances=_Lookups([(var, name) for var in uris for name in names], instance),
            cell_ref=cell_ref,
        )

    return of


# Each kind's fingerprint begins ``<name>:`` and then this.
_PREFIX = {"mount": "", "table": "table:", "fetch": "fetch:", "dataset": "dataset:"}


def _resolvers(fingerprints: Mapping[str, list[str]]) -> dict[str, cg.Resolver]:
    """Each input's fingerprint, found by its name among those already made."""

    def resolver(kind: str) -> cg.Resolver:
        def resolve(spec: cg.Input) -> cg.Resolved | cg.Unavailable:
            prefix = f"{spec.name}:{_PREFIX[kind]}"
            found = next((c for c in fingerprints[kind] if c.startswith(prefix)), None)
            if found is None:
                return cg.Unavailable(f"no fingerprint for {kind} {spec.name}")
            return cg.Resolved(identity=None, fingerprint=found)

        return resolve

    return {kind: resolver(kind) for kind in _PREFIX}


def input_parts(fingerprints: Mapping[str, list[str]]) -> list[list[str]]:
    """Each input as a key part, ``[kind, name, component]``, as cellgraph builds it.

    A fingerprint begins with its input's name.
    """
    parts = []
    for kind, found in fingerprints.items():
        for fingerprint in found:
            name = fingerprint.partition(":")[0]
            resolved = cg.Resolved(identity=None, fingerprint=fingerprint)
            parts.append([kind, name, cg.input_component(kind, name, resolved)])
    return sorted(parts)


def _label(value: list[str]) -> str:
    """A ``derive_subkey`` label for *value*, which holds no ':'."""
    return hashlib.sha256(cg.canonical_json(value).encode("utf-8")).hexdigest()


def inactive_variant_refusal(analysis: cg.Analysis, cell_id: str) -> str | None:
    """Why *cell_id* cannot run or be keyed, when it is an inactive variant.

    cellgraph neither wires nor keys one: what it reads depends on which member is
    active.
    """
    if cell_id not in analysis.graph.inactive_cells:
        return None
    member = analysis.cells[cell_id].member
    if member is None:
        return f"{cell_id!r} is an inactive variant; make it active to run it"
    return (
        f"{member.name!r} is not the active variant of group {member.group!r}; "
        "make it active to run it"
    )


def cell_key(
    session: ExecutorHost,
    analysis: cg.Analysis,
    cell_id: str,
    *,
    mounts: list[str],
    tables: list[str],
    fetches: list[str],
    datasets: list[str],
    env_hash: str,
    instance: str | cg.Item | None = None,
) -> CellKey:
    """*cell_id*'s provenance hash (one fan-out *instance*'s, when given) over these
    input fingerprints."""
    if analysis.structural_errors:
        # Without a DAG nothing is wired upstream, as before cellgraph.
        definition_key = analysis.cells[cell_id].definition_key
        assert definition_key is not None  # every cell has one
        digest = compute_provenance_hash(
            cell_provenance_inputs([], mounts, tables, fetches, datasets),
            definition_key,
            env_hash,
        )
        # A label may not hold ':', which an item key (a path, a URL) or a variant name
        # may: each is hashed in as canonical JSON.
        if isinstance(instance, cg.Item):
            digest = cg.derive_subkey(digest, _label(["each", instance.key, instance.digest]))
        elif instance is not None:
            digest = cg.derive_subkey(digest, _label(["variant", instance]))
        return CellKey(digest, [])
    result = analysis.execution_key(
        cell_id,
        instance=instance,
        published=published(session, analysis),
        resolve=_resolvers(
            {"mount": mounts, "table": tables, "fetch": fetches, "dataset": datasets}
        ),
        env_hash=env_hash,
    )
    if isinstance(result, cg.Unavailable):
        raise InputNotFingerprinted(f"cell {cell_id} has no key: {result.reason}")
    assert result.parts is not None
    upstream = [value for _address, value in result.parts["upstream"]]
    if isinstance(result, cg.Keyed):
        return CellKey(result.digest, upstream, result.parts)
    if result.unresolved:
        # A digest without what it could not read would match a record made without it.
        raise InputNotFingerprinted(f"cell {cell_id} has no key: {', '.join(result.unresolved)}")
    # A prompt or SQL cell's generic hash, which it records beside its own key. Its parts
    # have no inputs (they are the host's to key), so supply the ones this hash reads.
    inputs = input_parts({"mount": mounts, "table": tables, "fetch": fetches, "dataset": datasets})
    parts = {**result.parts, "inputs": inputs}
    return CellKey(cg.parts_digest(parts), upstream, parts)
