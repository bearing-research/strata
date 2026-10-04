"""Registry names as recorded cell inputs (``@dataset``).

``# @dataset model taxi/model@champion`` resolves the name, through the registry
the ambient client uses, to one ``id@v=N``; copies that version into the
notebook's store; binds it to ``model`` like an upstream variable; and folds
``"<var>:dataset:<reference>:<id>@v=<n>"`` into provenance. ``name@alias`` and a
bare name go stale when their pointer moves; ``name@v=N`` never does. A value the
harness can read is bound as that value; anything else arrives as a ``Path``.
The copy keeps its ``input_versions``, and the chain behind it is copied too, so
lineage in the notebook walks past the dataset to the steps that made it.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from typing import Any, Protocol
from urllib.parse import quote

import httpx

from strata.artifact_store import ArtifactStore, ArtifactVersion
from strata.artifact_transfer import remap_input_versions
from strata.notebook.models import DatasetSpec

STALE_CHECK_SECONDS = 60.0
LOOKUP_TIMEOUT_SECONDS = 10.0
DOWNLOAD_TIMEOUT_SECONDS = 300.0
# How many steps behind a dataset are copied with it, as the lineage walk reads.
ANCESTRY_DEPTH = 10

# Input URIs whose recorded version names an artifact the lineage walk follows.
_LINEAGE_PREFIXES = ("strata://artifact/", "strata://name/")

# The content types a cell's harness reads back into a value. Anything else is
# handed over as a file.
VALUE_CONTENT_TYPES = frozenset({"arrow/ipc", "json/object", "pickle/object"})


class DatasetError(RuntimeError):
    """A declared dataset could not be resolved or copied."""


@dataclass(frozen=True)
class ResolvedDataset:
    """What one ``@dataset`` resolved to in the registry."""

    spec: DatasetSpec
    artifact_id: str
    version: int

    @property
    def ref(self) -> str:
        return f"{self.artifact_id}@v={self.version}"

    @property
    def fingerprint(self) -> str:
        return f"{self.spec.name}:dataset:{self.spec.reference}:{self.ref}"

    @property
    def lineage_uri(self) -> str:
        """The ``strata://name/`` input URI an artifact records for this dataset."""
        return f"strata://name/{self.spec.reference}"


@dataclass(frozen=True)
class DatasetInput:
    """A resolved dataset, present in the notebook's store."""

    resolved: ResolvedDataset
    local_ref: str
    content_type: str


def unresolved_fingerprint(spec: DatasetSpec) -> str:
    """A fingerprint no artifact can match, so an unresolvable name is stale."""
    return f"{spec.name}:dataset:{spec.reference}:unresolved:{uuid.uuid4().hex}"


def content_type_of(record: ArtifactVersion) -> str:
    """Return *record*'s content type if the harness can read it, else ``file/path``.

    A record with no content type (written outside a notebook) is Arrow IPC.
    """
    content_type = "arrow/ipc"
    if record.transform_spec:
        try:
            params = json.loads(record.transform_spec).get("params") or {}
        except (ValueError, AttributeError):
            params = {}
        content_type = params.get("content_type") or content_type
    return content_type if content_type in VALUE_CONTENT_TYPES else "file/path"


class Registry(Protocol):
    def resolve(self, spec: DatasetSpec) -> ResolvedDataset: ...

    def download(self, artifact_id: str, version: int) -> tuple[ArtifactVersion, bytes] | None:
        """A readable version and its bytes, or ``None`` when the registry has none."""
        ...


class LocalRegistry:
    """The registry of a store on this machine."""

    def __init__(self, store: ArtifactStore):
        self._store = store

    def resolve(self, spec: DatasetSpec) -> ResolvedDataset:
        if spec.alias is not None:
            record = self._store.resolve_alias(spec.dataset, spec.alias)
        else:
            record = self._store.resolve_name(spec.dataset)
        if record is None:
            raise DatasetError(f"@dataset {spec.name}: {spec.reference} is not in the registry")
        if spec.version is not None:
            pinned = self._store.get_artifact(record.id, spec.version)
            if pinned is None:
                raise DatasetError(
                    f"@dataset {spec.name}: {spec.dataset} has no version {spec.version}"
                )
            record = pinned
        return ResolvedDataset(spec=spec, artifact_id=record.id, version=record.version)

    def download(self, artifact_id: str, version: int) -> tuple[ArtifactVersion, bytes] | None:
        record = self._store.get_artifact(artifact_id, version)
        if record is None or record.state not in ("ready", "superseded"):
            return None
        blob = self._store.read_blob(artifact_id, version)
        return None if blob is None else (record, blob)


class RemoteRegistry:
    """The registry of a store reached over HTTP."""

    def __init__(self, base_url: str, headers: dict[str, str] | None = None):
        self._base_url = base_url.rstrip("/")
        self._headers = dict(headers or {})

    def _get(self, path: str, timeout: float) -> httpx.Response:
        try:
            return httpx.get(f"{self._base_url}{path}", headers=self._headers, timeout=timeout)
        except httpx.HTTPError as exc:
            raise DatasetError(f"the registry at {self._base_url} is unreachable: {exc}") from exc

    def resolve(self, spec: DatasetSpec) -> ResolvedDataset:
        name = quote(spec.dataset, safe="/")
        if spec.alias is not None:
            path = f"/v1/names/{name}/aliases/{quote(spec.alias, safe='')}"
        else:
            path = f"/v1/names/{name}"
        response = self._get(path, LOOKUP_TIMEOUT_SECONDS)
        if response.status_code == 404:
            raise DatasetError(f"@dataset {spec.name}: {spec.reference} is not in the registry")
        if response.status_code >= 400:
            raise DatasetError(
                f"@dataset {spec.name}: the registry refused {spec.reference} "
                f"with HTTP {response.status_code}"
            )
        ref = str(response.json().get("artifact_uri") or "").removeprefix("strata://artifact/")
        artifact_id, _, version = ref.partition("@v=")
        if not artifact_id or not version.isdigit():
            raise DatasetError(
                f"@dataset {spec.name}: the registry answered {spec.reference} "
                "without an artifact reference"
            )
        if spec.version is None:
            return ResolvedDataset(spec=spec, artifact_id=artifact_id, version=int(version))
        if self._info(artifact_id, spec.version) is None:
            raise DatasetError(
                f"@dataset {spec.name}: {spec.dataset} has no version {spec.version}"
            )
        return ResolvedDataset(spec=spec, artifact_id=artifact_id, version=spec.version)

    def _info(self, artifact_id: str, version: int) -> dict[str, Any] | None:
        path = f"/v1/artifacts/{quote(artifact_id, safe='')}/v/{version}"
        response = self._get(path, LOOKUP_TIMEOUT_SECONDS)
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise DatasetError(
                f"the registry refused {artifact_id}@v={version} with HTTP {response.status_code}"
            )
        return response.json()

    def download(self, artifact_id: str, version: int) -> tuple[ArtifactVersion, bytes] | None:
        info = self._info(artifact_id, version)
        if info is None:
            return None
        path = f"/v1/artifacts/{quote(artifact_id, safe='')}/v/{version}/data"
        response = self._get(path, DOWNLOAD_TIMEOUT_SECONDS)
        if response.status_code == 404:
            return None
        if response.status_code >= 400 or not info.get("provenance_hash"):
            raise DatasetError(
                f"the bytes of {artifact_id}@v={version} could not be read "
                f"(HTTP {response.status_code})"
            )
        record = ArtifactVersion(
            id=artifact_id,
            version=version,
            state="ready",
            provenance_hash=str(info["provenance_hash"]),
            schema_json=info.get("arrow_schema"),
            row_count=info.get("row_count"),
            byte_size=info.get("byte_size"),
            created_at=info.get("created_at"),
            transform_spec=info.get("transform_spec"),
            input_versions=info.get("input_versions"),
        )
        return record, response.content


def registry_for(config: Any) -> Registry:
    """The registry ``@dataset`` reads: the remote store if configured, else this server's."""
    remote = getattr(config, "notebook_remote_store_url", None)
    if remote:
        from strata.auth import remote_store_headers

        return RemoteRegistry(str(remote), remote_store_headers(config))
    from strata.artifact_store import get_artifact_store

    store = get_artifact_store(getattr(config, "artifact_dir", None))
    if store is None:
        raise DatasetError(
            "@dataset needs a registry: set artifact_dir, or notebook_remote_store_url"
        )
    return LocalRegistry(store)


def copy_into(registry: Registry, resolved: ResolvedDataset, store: ArtifactStore) -> DatasetInput:
    """Copy *resolved* and the chain behind it into the notebook's *store*, if absent."""
    local = store.get_artifact(resolved.artifact_id, resolved.version)
    if local is not None and local.state in ("ready", "superseded"):
        return DatasetInput(
            resolved=resolved, local_ref=resolved.ref, content_type=content_type_of(local)
        )
    fetched = registry.download(resolved.artifact_id, resolved.version)
    if fetched is None:
        raise DatasetError(f"@dataset {resolved.spec.name}: {resolved.ref} has no stored bytes")
    record, blob = fetched
    landed = _land(registry, store, record, blob, depth=0, landed={})
    return DatasetInput(resolved=resolved, local_ref=landed, content_type=content_type_of(record))


def _land(
    registry: Registry,
    store: ArtifactStore,
    record: ArtifactVersion,
    blob: bytes,
    *,
    depth: int,
    landed: dict[str, str],
) -> str:
    """Import *record* after its ancestors; return the ref it landed on.

    An ancestor the registry cannot hand over stays a dangling edge, as it is in
    the registry's own lineage. An edge is rewritten when its ancestor landed on a
    row that already held the same computation.
    """
    edges = json.loads(record.input_versions) if record.input_versions else {}
    if depth >= ANCESTRY_DEPTH:
        edges = {}
    for uri, value in edges.items():
        ref = str(value)
        artifact_id, sep, version = ref.partition("@v=")
        if not uri.startswith(_LINEAGE_PREFIXES) or not sep or not version.isdigit():
            continue
        if ref in landed:
            continue
        local = store.get_artifact(artifact_id, int(version))
        if local is not None and local.state in ("ready", "superseded"):
            landed[ref] = ref
            continue
        fetched = registry.download(artifact_id, int(version))
        if fetched is None:
            continue
        landed[ref] = _land(registry, store, *fetched, depth=depth + 1, landed=landed)
    # The notebook's store is untenanted.
    record = remap_input_versions(replace(record, tenant=None), landed)
    return store.import_artifact(record, blob).ref
