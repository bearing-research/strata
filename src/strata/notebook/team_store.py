"""Consult a shared team store when the local cache misses.

The notebook provenance key holds no notebook or cell id, so two people running
the same source over the same inputs in the same environment get the same hash.

Pull-through, not redirection: the store is asked only on a local miss, and
what comes back is written into the local store under the local canonical id,
so later reads are local and a store outage never breaks a notebook.

Nothing here can fail a cell. Every failure returns "no result" and logs; a
genuine miss (marked by :data:`PROVENANCE_MISS_HEADER`) and an outage differ
only in log level.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import anyio
import httpx
from vestigo.provenance import derive_subkey

from strata.notebook import output_dirs
from strata.notebook.artifact_integration import NotebookArtifactManager
from strata.observability.logging import get_logger
from strata.types import PROVENANCE_MISS_HEADER

logger = get_logger(__name__)

# A lookup is on every cell run's miss path, so it gets a short leash (recomputing is
# the fallback anyway). The download is longer: its size is not ours to predict.
LOOKUP_TIMEOUT_SECONDS = 10.0
DOWNLOAD_TIMEOUT_SECONDS = 300.0
# A publish runs after the cell finished, but the user is waiting to see output, so
# it cannot be open-ended.
PUBLISH_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class TeamArtifact:
    """One variable's result, fetched from the shared store."""

    provenance_hash: str
    artifact_id: str
    version: int
    content_type: str
    principal: str | None
    blob: bytes
    build_env: str = ""
    build_duration_ms: int = 0
    env_hash: str = ""
    promotion: str | None = None


@dataclass(frozen=True)
class TeamPull:
    """A completed pull: every consumed variable landed in the local store.

    ``principal`` is who computed it; the executor surfaces it so a hit reads as
    "alice already ran this" rather than an unexplained instant success.
    """

    variables: tuple[str, ...]
    principal: str | None
    byte_size: int
    # Where the result was computed, e.g. ``cpython-3.14-linux-x86_64``. The provenance
    # key covers the lockfile, not the platform, so a hit can cross machines; this records
    # which. Empty when the producer reported none.
    build_env: str = ""
    # The publisher's run cost, i.e. what this hit saved. The puller has no history for a
    # cell they never ran.
    saved_ms: int = 0
    # The promotion whose chain put the result in the store; ``None`` when it arrived by
    # a cache publish.
    promotion: str | None = None


class TeamStore:
    """The shared artifact store, as the notebook talks to it."""

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ):
        """Create a client for the shared store at ``base_url``.

        ``headers`` (trusted-proxy identity, tenant, scopes) are sent per request, so
        an injected ``client`` cannot silently drop authentication. An injected
        ``client`` is owned by the caller and not closed here.
        """
        # Absolute URLs, not the client's ``base_url``, so an injected client cannot send
        # relative paths nowhere.
        self._base_url = base_url.rstrip("/")
        self._headers = dict(headers or {})
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def fetch(self, provenance_hash: str) -> TeamArtifact | None:
        """Look up ``provenance_hash`` and download the bytes if anyone computed it.

        Returns ``None`` for a miss and for every failure; the log tells them apart.
        """
        match = await self._lookup(provenance_hash)
        if match is None:
            return None

        artifact_id = match.get("artifact_id")
        version = match.get("version")
        if not artifact_id or version is None:
            logger.warning(
                "Team store answered %s without an artifact reference",
                provenance_hash[:12],
            )
            return None

        blob = await self._download(str(artifact_id), int(version))
        if blob is None:
            return None

        return TeamArtifact(
            provenance_hash=provenance_hash,
            artifact_id=str(artifact_id),
            version=int(version),
            content_type=str(match.get("content_type") or ""),
            principal=match.get("principal"),
            blob=blob,
            build_env=str(match.get("build_env") or ""),
            build_duration_ms=int(match.get("build_duration_ms") or 0),
            env_hash=str(match.get("env_hash") or ""),
            promotion=match.get("promotion") or None,
        )

    async def publish(
        self,
        provenance_hash: str,
        blob: bytes | BinaryIO,
        *,
        content_type: str,
        variable_name: str | None = None,
        build_env: str = "",
        build_duration_ms: int = 0,
        env_hash: str = "",
    ) -> tuple[str, int] | None:
        """Offer a result to the team, keyed by provenance. Never raises.

        *blob* is bytes or an open file, sent as it is read. Returns the ``(id, version)``
        it landed on, or ``None``: a failed push costs the next person a recompute and must
        not fail a cell that ran fine.
        """
        metadata: dict[str, str] = {"content_type": content_type}
        if variable_name:
            metadata["variable_name"] = variable_name
        if build_env:
            metadata["build_env"] = build_env
        if build_duration_ms > 0:
            metadata["build_duration_ms"] = str(build_duration_ms)
        # Package-set half of the environment identity; the platform half rides in build_env.
        # Both travel so "you got a hit and I did not" can be answered.
        if env_hash:
            metadata["env_hash"] = env_hash
        files = {
            "metadata": ("metadata.json", json.dumps(metadata), "application/json"),
            "data": ("data.bin", blob, "application/octet-stream"),
        }
        try:
            response = await self._client.put(
                f"{self._base_url}/v1/artifacts/by-provenance/{provenance_hash}",
                files=files,
                headers=self._headers,
                timeout=PUBLISH_TIMEOUT_SECONDS,
            )
        except (httpx.HTTPError, OSError) as exc:
            logger.warning(
                "Team store unreachable publishing %s (%s); the result stays local",
                provenance_hash[:12],
                exc,
            )
            return None

        if response.status_code >= 400:
            # 403 is usually a read-only member: legitimate, but visible, or a team wonders why
            # nothing is ever shared.
            logger.warning(
                "Team store refused to publish %s with HTTP %d; the result stays local",
                provenance_hash[:12],
                response.status_code,
            )
            return None
        try:
            uri = str(response.json()["artifact_uri"])
            landed_id, _, landed_version = uri.removeprefix("strata://artifact/").rpartition("@v=")
            return landed_id, int(landed_version)
        except (ValueError, KeyError, TypeError):
            logger.warning("Team store took %s but did not say where", provenance_hash[:12])
            return None

    async def holds(self, provenance_hash: str, sha256: str) -> tuple[str, int] | None:
        """Where the store keeps these exact bytes under *provenance_hash*, if it does."""
        match = await self._lookup(provenance_hash)
        if match is None or match.get("content_sha256") != sha256:
            return None
        try:
            return str(match["artifact_id"]), int(match["version"])
        except (KeyError, TypeError, ValueError):
            return None

    async def download_to(
        self, artifact_id: str, version: int, dest: Path
    ) -> tuple[str, int] | None:
        """Write a version's bytes to *dest* as they arrive; their sha256 and size, or ``None``."""
        digest = hashlib.sha256()
        size = 0
        try:
            async with self._client.stream(
                "GET",
                f"{self._base_url}/v1/artifacts/{artifact_id}/v/{version}/data",
                headers=self._headers,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            ) as response:
                response.raise_for_status()
                with open(dest, "wb") as out:
                    async for chunk in response.aiter_bytes():
                        digest.update(chunk)
                        size += len(chunk)
                        out.write(chunk)
        except (httpx.HTTPError, OSError) as exc:
            logger.warning(
                "Team store named %s@v=%d but it could not be fetched: %s",
                artifact_id,
                version,
                exc,
            )
            return None
        return digest.hexdigest(), size

    async def _lookup(self, provenance_hash: str) -> dict | None:
        try:
            response = await self._client.get(
                f"{self._base_url}/v1/artifacts/by-provenance/{provenance_hash}",
                headers=self._headers,
                timeout=LOOKUP_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "Team store unreachable looking up %s (%s); recomputing locally",
                provenance_hash[:12],
                exc,
            )
            return None

        if response.status_code == 404 and PROVENANCE_MISS_HEADER in response.headers:
            logger.debug("Team store has no result for %s", provenance_hash[:12])
            return None
        if response.status_code >= 400:
            # Warn, not debug: an unmarked 404 (old store, no artifacts) or a 403 (expired token)
            # looks exactly like an empty cache and would be an unexplained permanent slowdown.
            logger.warning(
                "Team store refused a lookup of %s with HTTP %d; recomputing locally",
                provenance_hash[:12],
                response.status_code,
            )
            return None

        try:
            payload = response.json()
        except ValueError:
            logger.warning("Team store returned a non-JSON provenance match")
            return None
        return payload if isinstance(payload, dict) else None

    async def _download(self, artifact_id: str, version: int) -> bytes | None:
        """The raw stored bytes, not an Arrow table.

        Only the serializer knows a variable's content type (arrow, JSON, pickle), so
        the bytes stay opaque end to end.
        """
        try:
            response = await self._client.get(
                f"{self._base_url}/v1/artifacts/{artifact_id}/v/{version}/data",
                headers=self._headers,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            # The store claimed the result but can't serve it: inconsistent, not an empty cache.
            logger.warning(
                "Team store matched %s@v=%d but its bytes could not be fetched: %s",
                artifact_id,
                version,
                exc,
            )
            return None
        return response.content


async def pull_cell_outputs(
    store: TeamStore,
    artifact_mgr: NotebookArtifactManager,
    *,
    cell_id: str,
    provenance_hash: str,
    consumed_vars: set[str],
    source_hash: str = "",
    source: str = "",
    env_hash: str = "",
    input_versions: dict[str, str] | None = None,
    variant: str | None = None,
) -> TeamPull | None:
    """Materialise a teammate's result for this cell in the local store.

    All or nothing: the cache-hit check needs every consumed variable, so all are
    fetched before any is written, and the pull stops at the first absent one. A
    download failing after a write leaves correctly keyed rows behind; the hit
    check keeps rejecting the incomplete set. Returns ``None`` on any miss or failure.

    An output folder comes with every file its manifest names, each checked against
    its entry, and lands under this notebook's own ids before its manifest does.
    """
    if not consumed_vars:
        # Nothing downstream means no artifact; console and display outputs live in runtime
        # state, not the store.
        return None

    ordered = sorted(consumed_vars)
    fetched = await asyncio.gather(
        *(store.fetch(derive_subkey(provenance_hash, var)) for var in ordered)
    )

    missing = [var for var, artifact in zip(ordered, fetched, strict=True) if artifact is None]
    if missing:
        logger.debug(
            "Team store is missing %s for cell %s; running it locally",
            ", ".join(missing),
            cell_id,
        )
        return None

    pulled = [
        (var, artifact)
        for var, artifact in zip(ordered, fetched, strict=True)
        if artifact is not None
    ]

    # The publisher's env_hash, kept rather than restamped. Provenance folds in env_hash,
    # so an honest pull cannot disagree. If it does, importing it would make
    # `causality._get_stored_hash` report "the environment changed" forever with no
    # visible cause, so keep the local value and warn. Check every variable, not just the
    # first, or a bad env_hash on a later one is silently laundered.
    disputed = sorted(
        {
            artifact.env_hash
            for _, artifact in pulled
            if artifact.env_hash and env_hash and artifact.env_hash != env_hash
        }
    )
    published_env_hash = pulled[0][1].env_hash
    if disputed:
        logger.warning(
            "Team store returned cell %s with env_hash %s but its provenance was "
            "computed under %s; these cannot both be right. Keeping the local "
            "value; treat the store's metadata as suspect.",
            cell_id,
            ", ".join(h[:12] for h in disputed),
            env_hash[:12],
        )
        published_env_hash = ""

    with (
        tempfile.TemporaryDirectory(prefix="strata_team_pull_") as workdir,
        output_dirs.assembling(),
    ):
        # Every file of every folder is fetched and checked before anything lands, so no
        # manifest lands naming a file that is not here.
        fetched_files: dict[str, dict[str, tuple[Path, str, int]]] = {}
        for var, artifact in pulled:
            if artifact.content_type == output_dirs.CONTENT_TYPE:
                files = await _fetch_folder_files(store, artifact, Path(workdir) / var)
                if files is None:
                    return None
                fetched_files[var] = files
        manifests = await anyio.to_thread.run_sync(
            _land_folder_files, artifact_mgr, cell_id, fetched_files
        )
        total_bytes = sum(size for files in fetched_files.values() for _, _, size in files.values())
        for var, artifact in pulled:
            artifact_mgr.store_cell_output(
                cell_id=cell_id,
                variable_name=var,
                blob_data=manifests.get(var, artifact.blob),
                content_type=artifact.content_type,
                provenance_hash=artifact.provenance_hash,
                # Same upstream refs a local run records, so lineage doesn't depend on which cells
                # happened to hit the shared cache.
                input_versions=input_versions or {},
                source_hash=source_hash,
                # The local source, not the publisher's: a provenance match only proves AST
                # identity, and annotations (``# @worker``, ``# @env``) are comments that can
                # differ. Record what this machine would have run.
                source=source,
                env_hash=published_env_hash or env_hash,
                variant=variant,
                # Preserved, not restamped: the bytes came from the publisher's machine.
                build_env=artifact.build_env,
                build_duration_ms=artifact.build_duration_ms,
                # Persisted, not just logged: lineage reads the local store, so otherwise the author
                # column is blank on exactly the steps someone else produced.
                principal=artifact.principal,
            )
            total_bytes += len(artifact.blob)

    # One cell run computed every variable, so they share an author.
    principal = pulled[0][1].principal
    build_env = pulled[0][1].build_env
    # Cost of the run, not a sum over variables (which would multiply it).
    saved_ms = max(artifact.build_duration_ms for _, artifact in pulled)
    # Variables can arrive separately (one promoted, one cache-published). It is a
    # promotion hit if any was; min() keeps the reported one stable.
    promotion = min((a.promotion for _, a in pulled if a.promotion), default=None)
    logger.info(
        "Team store supplied cell %s (%s, %d bytes, computed by %s on %s); skipped running it",
        cell_id,
        ", ".join(ordered),
        total_bytes,
        principal or "an unrecorded author",
        build_env or "an unrecorded platform",
    )
    return TeamPull(
        variables=tuple(ordered),
        principal=principal,
        byte_size=total_bytes,
        build_env=build_env,
        saved_ms=saved_ms,
        promotion=promotion,
    )


async def publish_cell_outputs(
    store: TeamStore,
    artifact_mgr: NotebookArtifactManager,
    *,
    cell_id: str,
    consumed_vars: set[str],
    variant: str | None = None,
) -> int:
    """Offer this cell's freshly computed outputs to the team.

    Reads back the *local* store's artifacts (not the harness temp files), so what
    is published is exactly what a pull must reproduce, with the stored provenance
    key. Not all-or-nothing: a partial publish just means the next person
    recomputes. Returns how many variables landed. Never raises.
    """
    published = 0
    for var in sorted(consumed_vars):
        artifact_id = artifact_mgr.cell_artifact_id(cell_id, var, variant=variant)
        stored = artifact_mgr.artifact_store.get_latest_version(artifact_id)
        if stored is None or stored.state not in ("ready", "superseded"):
            continue
        content_type = _spec_param(stored, "content_type")
        blob = artifact_mgr.artifact_store.read_blob(artifact_id, stored.version)
        if blob is None:
            continue
        if content_type == output_dirs.CONTENT_TYPE:
            blob = await _offer_folder_files(store, artifact_mgr, stored.id, blob)
            if blob is None:
                continue
        if await store.publish(
            stored.provenance_hash,
            blob,
            content_type=content_type,
            variable_name=var,
            build_env=_spec_param(stored, "build_env"),
            build_duration_ms=_spec_int(stored, "build_duration_ms"),
            env_hash=_spec_param(stored, "env_hash"),
        ):
            published += 1

    if published:
        logger.info(
            "Published %d of cell %s's outputs to the team store",
            published,
            cell_id,
        )
    return published


async def _fetch_folder_files(
    store: TeamStore, artifact: TeamArtifact, workdir: Path
) -> dict[str, tuple[Path, str, int]] | None:
    """Download each file a pulled folder names into *workdir*: ``{path: (file, sha256, size)}``.

    ``None`` when the manifest cannot be read, or any file is missing or differs from its
    entry: the manifest came from another store, so nothing in it is taken on trust.
    """
    try:
        described = json.loads(artifact.blob)
        named = [
            (
                output_dirs.checked_path(rel),
                *output_dirs.file_ref(artifact.artifact_id, described["cell_id"], entry),
                str(entry["sha256"]),
                int(entry["size"]),
            )
            for rel, entry in described["files"].items()
        ]
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        logger.warning(
            "Team store's folder %s@v=%d cannot be read (%s); running it locally",
            artifact.artifact_id,
            artifact.version,
            exc,
        )
        return None
    workdir.mkdir()
    files: dict[str, tuple[Path, str, int]] = {}
    for index, (rel, file_id, version, sha256, size) in enumerate(named):
        dest = workdir / str(index)
        got = await store.download_to(file_id, version, dest)
        if got is None:
            return None
        if got != (sha256, size):
            logger.warning(
                "Team store's %s@v=%d is not the file %s its folder names; running it locally",
                file_id,
                version,
                rel,
            )
            return None
        files[rel] = (dest, sha256, size)
    return files


def _land_folder_files(
    artifact_mgr: NotebookArtifactManager,
    cell_id: str,
    fetched: dict[str, dict[str, tuple[Path, str, int]]],
) -> dict[str, bytes]:
    """Store each fetched file under this notebook's id; return each folder's manifest.

    The manifest is the one a run here would have collected, so the folder rebuilds
    from this store alone.
    """
    return {
        var: output_dirs.collect(
            artifact_mgr,
            cell_id,
            None,
            var,
            {
                rel: output_dirs.store_received(artifact_mgr, cell_id, path, sha256, size)
                for rel, (path, sha256, size) in files.items()
            },
        )
        for var, files in fetched.items()
    }


async def _offer_folder_files(
    store: TeamStore,
    artifact_mgr: NotebookArtifactManager,
    manifest_id: str,
    manifest: bytes,
) -> bytes | None:
    """Offer each file a folder names; return the manifest naming where each landed.

    ``None``, and nothing offered, when a file is not here to send; ``None`` too when one
    does not land, so the manifest is never published without its files.
    """
    local = artifact_mgr.artifact_store
    records = []
    for file_id, version, sha256 in output_dirs.file_refs(manifest_id, manifest):
        record = local.get_artifact(file_id, version)
        if record is None or record.state not in ("ready", "superseded"):
            logger.warning(
                "Not offering folder %s to the team store: it names %s@v=%d, which is gone",
                manifest_id,
                file_id,
                version,
            )
            return None
        records.append((record, sha256))

    landed: dict[tuple[str, int], tuple[str, int]] = {}
    for record, sha256 in records:
        # A file unchanged since an earlier run is there already; it is not sent again.
        held = await store.holds(record.provenance_hash, sha256)
        if held is None:
            reader = local.open_blob_reader(record.id, record.version)
            if reader is None:
                return None
            with reader as handle:
                held = await store.publish(
                    record.provenance_hash,
                    handle,
                    content_type=_spec_param(record, "content_type"),
                )
            if held is None:
                return None
        landed[(record.id, record.version)] = held
    # The store mints the manifest's id, so no file id can be read off it.
    return output_dirs.relocate(manifest, manifest_id, landed, every_id=True)


def _spec_param(artifact, key: str) -> str:
    """One value from the stored transform spec's params, or empty.

    Used for ``content_type`` and ``build_env``. Empty rather than a guess: a
    wrong default content type would decode a pickle incorrectly.
    """
    return str(artifact.params.get(key) or "")


def _spec_int(artifact, key: str) -> int:
    """Like :func:`_spec_param`, for an integer param (milliseconds)."""
    try:
        return int(_spec_param(artifact, key))
    except ValueError:
        return 0
