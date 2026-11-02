"""A cell's execution key, from cellgraph.

cellgraph assembles what a cell reads upstream and keys Python and R cells. Strata
says what each upstream published (its artifacts) and what each input resolved to
(the fingerprints staleness and the executor already make). A prompt or SQL cell
keys itself; its generic hash, recorded beside that key, is made here over the same
upstreams.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cellgraph as cg

from strata.notebook.provenance import cell_provenance_inputs, compute_provenance_hash

if TYPE_CHECKING:
    from strata.notebook.session import NotebookSession


class InputNotFingerprinted(RuntimeError):
    """An input the cell declares is missing from the fingerprints it was keyed over."""


@dataclass(frozen=True)
class CellKey:
    """A run's provenance hash, and the upstream components it covers."""

    provenance_hash: str
    upstream: list[str]


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


def published(session: NotebookSession, analysis: cg.Analysis) -> Callable[[str], cg.Published]:
    """What each cell published, as cellgraph reads it: its artifacts in the store."""
    manager = session.artifact_manager
    store = manager.artifact_store

    def ref(artifact: Any, by_content: bool) -> cg.OutputRef:
        digest = None
        if by_content:
            # Recorded at finalize; backfilled once if absent.
            digest = artifact.content_sha256 or store.content_digest(artifact.id, artifact.version)
        return cg.OutputRef(cg.Provenance(artifact.provenance_hash), digest)

    def from_uri(uri: str, by_content: bool) -> cg.OutputRef | cg.Unavailable:
        try:
            artifact_id, version = session._parse_artifact_uri(uri)
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
            return cg.Resolved(identity=None, fingerprint=None, legacy=found)

        return resolve

    return {kind: resolver(kind) for kind in _PREFIX}


def cell_key(
    session: NotebookSession,
    analysis: cg.Analysis,
    cell_id: str,
    *,
    mounts: list[str],
    tables: list[str],
    fetches: list[str],
    datasets: list[str],
    env_hash: str,
    instance: str | None = None,
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
        if instance is not None:
            digest = cg.derive_subkey(digest, f"variant={instance}")
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
        return CellKey(result.digest, upstream)
    # A prompt or SQL cell's generic hash, which it records beside its own key.
    digest = compute_provenance_hash(
        cell_provenance_inputs(upstream, mounts, tables, fetches, datasets),
        result.parts["definition_key"],
        env_hash,
    )
    if instance is not None:
        digest = cg.derive_subkey(digest, f"variant={instance}")
    return CellKey(digest, upstream)
