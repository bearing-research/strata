"""Registry names as recorded cell inputs (``@dataset``).

A cell that resolves a promoted name with the ambient ``strata`` client has the
name in its source hash and the version in nothing: a new champion does not make it
stale, and its artifacts do not say what they were computed from. ``@dataset``
makes the version an input::

    # @dataset model taxi/model@champion
    predictions = model.predict(rows)

The name resolves through the registry the notebook's ambient client uses (the
server's own store, or ``notebook_remote_store_url`` when one is set) to one
``id@v=N``. That version is copied into the notebook's store keeping its id and
version, bound to ``model`` like an upstream variable, recorded among the
artifact's inputs, and folded into provenance as
``"<var>:dataset:<reference>:<id>@v=<n>"``.

``name@alias`` follows the alias, so the cell goes stale when it moves; a bare
name follows the name pointer the same way; ``name@v=N`` pins a version of the
artifact the name points at, and a pin never goes stale.

A value the notebook knows how to read (Arrow, JSON, pickle) is bound as that
value. Anything else arrives as a ``Path`` to its bytes, as ``@fetch`` does.
Artifacts written outside a notebook carry no content type; their bytes are
the Arrow IPC every core transform produces.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from typing import Any, Protocol
from urllib.parse import quote

import httpx

from strata.artifact_store import ArtifactStore, ArtifactVersion
from strata.notebook.models import DatasetSpec

STALE_CHECK_SECONDS = 60.0
LOOKUP_TIMEOUT_SECONDS = 10.0
DOWNLOAD_TIMEOUT_SECONDS = 300.0

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
        """The input URI an artifact records; the lineage walk follows
        ``strata://name/`` inputs to the ``id@v=N`` they name."""
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
    """How a cell should receive *record*: its stored content type when the
    harness can read it back, the bytes as a file otherwise."""
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

    def download(self, resolved: ResolvedDataset) -> tuple[ArtifactVersion, bytes]: ...


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

    def download(self, resolved: ResolvedDataset) -> tuple[ArtifactVersion, bytes]:
        record = self._store.get_artifact(resolved.artifact_id, resolved.version)
        blob = self._store.read_blob(resolved.artifact_id, resolved.version)
        if record is None or blob is None:
            raise DatasetError(f"@dataset {resolved.spec.name}: {resolved.ref} has no stored bytes")
        return record, blob


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
        pinned = ResolvedDataset(spec=spec, artifact_id=artifact_id, version=spec.version)
        self._info(pinned)
        return pinned

    def _info(self, resolved: ResolvedDataset) -> dict[str, Any]:
        path = f"/v1/artifacts/{quote(resolved.artifact_id, safe='')}/v/{resolved.version}"
        response = self._get(path, LOOKUP_TIMEOUT_SECONDS)
        if response.status_code == 404:
            raise DatasetError(
                f"@dataset {resolved.spec.name}: {resolved.spec.dataset} has no version "
                f"{resolved.version}"
            )
        if response.status_code >= 400:
            raise DatasetError(
                f"@dataset {resolved.spec.name}: the registry refused {resolved.ref} "
                f"with HTTP {response.status_code}"
            )
        return response.json()

    def download(self, resolved: ResolvedDataset) -> tuple[ArtifactVersion, bytes]:
        info = self._info(resolved)
        path = f"/v1/artifacts/{quote(resolved.artifact_id, safe='')}/v/{resolved.version}/data"
        response = self._get(path, DOWNLOAD_TIMEOUT_SECONDS)
        if response.status_code >= 400 or not info.get("provenance_hash"):
            raise DatasetError(
                f"@dataset {resolved.spec.name}: the bytes of {resolved.ref} could not be "
                f"read (HTTP {response.status_code})"
            )
        record = ArtifactVersion(
            id=resolved.artifact_id,
            version=resolved.version,
            state="ready",
            provenance_hash=str(info["provenance_hash"]),
            schema_json=info.get("arrow_schema"),
            row_count=info.get("row_count"),
            byte_size=info.get("byte_size"),
            created_at=info.get("created_at"),
            transform_spec=info.get("transform_spec"),
        )
        return record, response.content


def registry_for(config: Any) -> Registry:
    """The registry a notebook's ``@dataset`` reads: the remote store when one is
    configured, as for the ambient client, else this server's own store."""
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
    """Make *resolved* readable from the notebook's *store*, keeping its id and
    version. A version already there is not downloaded again."""
    local = store.get_artifact(resolved.artifact_id, resolved.version)
    if local is not None and local.state in ("ready", "superseded"):
        return DatasetInput(
            resolved=resolved, local_ref=resolved.ref, content_type=content_type_of(local)
        )
    record, blob = registry.download(resolved)
    # The notebook's store is untenanted, and the ancestors the registry's
    # record names are not in it: the copy is a leaf here.
    landed = store.import_artifact(replace(record, tenant=None, input_versions=None), blob)
    return DatasetInput(
        resolved=resolved, local_ref=landed.ref, content_type=content_type_of(record)
    )
