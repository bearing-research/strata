"""Consult a shared store when the local cache misses.

The team cache hit: a colleague ran this exact computation, so you get their
result instead of spending the minutes again. It works because the notebook's
provenance key — ``sha256(sorted_input_hashes + source_hash + env_hash)`` —
contains no notebook id and no cell id. Two people running the same source over
the same inputs in the same environment already arrive at the same hash. Only
the lookup was missing.

**Pull-through, not redirection.** The notebook keeps its own local store and
asks the team store only on a miss; what comes back is written into the local
store under the local canonical id, so every later read is local and the
existing cache-hit validation passes without an exception carved into it. The
alternative — pointing the notebook's whole artifact store at the remote — would
put a network call behind every inter-cell variable read and turn a store
outage into a broken notebook.

**Nothing here can fail a cell.** A team store that is unreachable, slow, or
refusing is a store you recompute past. Every failure returns "no result" and
logs; the difference between "nobody computed this" and "I could not ask" is
kept in the *log level*, not in the return value, because a cell has to run
either way. That distinction only exists at all because the route marks a
genuine miss with :data:`PROVENANCE_MISS_HEADER` — without it an outage would
be indistinguishable from an empty cache, and a team would conclude the shared
cache does not work.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import httpx

from strata.logging import get_logger
from strata.notebook.artifact_integration import NotebookArtifactManager
from strata.notebook.provenance import derive_subkey
from strata.types import PROVENANCE_MISS_HEADER

logger = get_logger(__name__)

# A lookup sits on the miss path of every cell run, so it gets a short leash:
# waiting on an unreachable store is strictly worse than recomputing, which is
# the fallback anyway. The download gets a longer one because it is moving the
# result the user actually wants and its size is not ours to predict.
LOOKUP_TIMEOUT_SECONDS = 10.0
DOWNLOAD_TIMEOUT_SECONDS = 300.0
# A publish runs after the cell has already finished, so the user is waiting on
# nothing — but they are waiting to see their output, so it cannot be
# open-ended either.
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


@dataclass(frozen=True)
class TeamPull:
    """A completed pull: every consumed variable landed in the local store.

    ``principal`` is who computed it, and is the reason this is a distinct type
    rather than a bool. A result that materialises with no author is
    indistinguishable from a bug; the executor surfaces the name so a hit reads
    as "alice already ran this" rather than as an unexplained instant success.
    """

    variables: tuple[str, ...]
    principal: str | None
    byte_size: int
    # Where the result was computed, e.g. ``cpython-3.14-linux-x86_64``. The
    # provenance key covers the lockfile, not the platform, so a hit can cross
    # machines — recording which one it crossed from is what keeps that
    # honest rather than silent. Empty when the producer reported none.
    build_env: str = ""
    # What the publisher's run cost, and therefore what this hit saved. The
    # puller has no history for a cell they never ran, so without this the
    # savings estimate credits zero for exactly the case worth counting.
    saved_ms: int = 0


class TeamStore:
    """The shared artifact store, as the notebook talks to it."""

    def __init__(
        self,
        base_url: str,
        headers: dict[str, str] | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ):
        """
        Args:
            base_url: The shared store's root, e.g. ``https://store.example``.
            headers: Auth the store needs — trusted-proxy identity, tenant, and
                scopes. Sent per request rather than baked into an injected
                client, so handing in a client for tests or a proxy cannot
                silently drop authentication.
            client: Injected transport. When supplied it is not closed here;
                the caller owns what it created.
        """
        # Requests are built absolute rather than leaning on the client's
        # ``base_url``, so an injected client cannot silently send relative
        # paths nowhere. Where to reach the store is this object's business.
        self._base_url = base_url.rstrip("/")
        self._headers = dict(headers or {})
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def fetch(self, provenance_hash: str) -> TeamArtifact | None:
        """The whole round trip: has anyone computed this, and if so, the bytes.

        Returns ``None`` for a miss and for every failure — see the module
        docstring. The two are distinguished in the log, not in the result.
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
        )

    async def publish(
        self,
        provenance_hash: str,
        blob: bytes,
        *,
        content_type: str,
        variable_name: str | None = None,
        build_env: str = "",
        build_duration_ms: int = 0,
        env_hash: str = "",
    ) -> bool:
        """Offer a result to the team, keyed by provenance. Never raises.

        Returns whether it landed, which is used for logging and nothing else:
        a push that fails costs the *next* person a recomputation, and costs
        this one nothing. It must not turn a cell that ran fine into a cell
        that errored.
        """
        metadata: dict[str, str] = {"content_type": content_type}
        if variable_name:
            metadata["variable_name"] = variable_name
        if build_env:
            metadata["build_env"] = build_env
        if build_duration_ms > 0:
            metadata["build_duration_ms"] = str(build_duration_ms)
        # Half of the environment identity — which package set. The other half,
        # the platform, rides in build_env. Both travel because "you got a hit
        # and I did not" is answered by comparing them and by nothing else.
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
        except httpx.HTTPError as exc:
            logger.warning(
                "Team store unreachable publishing %s (%s); the result stays local",
                provenance_hash[:12],
                exc,
            )
            return False

        if response.status_code >= 400:
            # 403 here is the common and important one: a read-only member.
            # That is a legitimate configuration, not a fault, but it should be
            # visible — otherwise a team wonders why nothing is ever shared.
            logger.warning(
                "Team store refused to publish %s with HTTP %d; the result stays local",
                provenance_hash[:12],
                response.status_code,
            )
            return False
        return True

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
            # Deliberately warn rather than debug. An unmarked 404 is a store
            # that predates the route or has no artifacts configured; a 403 is
            # a token that has expired. All three look exactly like an empty
            # cache from the outside, and would otherwise be an unexplained
            # permanent slowdown.
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
        """The raw stored bytes — not an Arrow table.

        A cell variable can be any of the notebook's content types (arrow, JSON,
        pickle), and only the serializer knows which. Decoding here would force
        this module to learn all of them and re-encode before writing the blob
        back, so it stays opaque bytes end to end.
        """
        try:
            response = await self._client.get(
                f"{self._base_url}/v1/artifacts/{artifact_id}/v/{version}/data",
                headers=self._headers,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            # A match that cannot be downloaded is the one case worth flagging
            # loudly: the store said it had the result, so this is not an empty
            # cache but a store that is answering inconsistently.
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
    env_hash: str = "",
    variant: str | None = None,
) -> TeamPull | None:
    """Materialise a teammate's result for this cell in the local store.

    All or nothing. The executor's cache-hit check requires *every* consumed
    variable to have a local canonical artifact whose provenance matches, so a
    partial pull is a miss that also wrote rows — this fetches all of them
    before writing any, and gives up as soon as one is absent.

    (A download that fails after the first write leaves the earlier variables
    stored. That is benign: each is written under its own correct provenance,
    so the next attempt re-fetches only what is still missing and the cache-hit
    check keeps rejecting the incomplete set until it is complete.)

    Returns the pull on success, ``None`` on any miss or failure.
    """
    if not consumed_vars:
        # A cell with nothing downstream has no artifact to share. Its cached
        # console and display outputs live in the session's runtime state, not
        # in the artifact store, so there is nothing here to pull.
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

    # The publisher's environment identity, kept rather than restamped — same
    # reason their platform is. An honest pull cannot disagree: env_hash is
    # part of the provenance key, so a matching provenance implies a matching
    # env_hash.
    #
    # If they disagree anyway, something upstream is wrong, and importing the
    # value would push that wrongness somewhere it cannot be read as a problem:
    # `causality._get_stored_hash` prefers the stored env_hash over the cell's
    # own when explaining staleness, so the cell would report "the environment
    # changed" forever, with nothing pointing at why. Keep the local value and
    # say so out loud instead — a signal is only a signal if it surfaces as one.
    # Every variable, not just the first: they are written from one value, so
    # checking one and stamping all of them would silently launder a bad
    # env_hash on any variable that sorted later.
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
            "value — treat the store's metadata as suspect.",
            cell_id,
            ", ".join(h[:12] for h in disputed),
            env_hash[:12],
        )
        published_env_hash = ""

    total_bytes = 0
    for var, artifact in pulled:
        artifact_mgr.store_cell_output(
            cell_id=cell_id,
            variable_name=var,
            blob_data=artifact.blob,
            content_type=artifact.content_type,
            provenance_hash=artifact.provenance_hash,
            source_hash=source_hash,
            env_hash=published_env_hash or env_hash,
            variant=variant,
            # Preserved, not restamped. The bytes were produced on the
            # publisher's machine, so the local copy has to keep saying so —
            # overwriting it with this machine's identity would turn a record
            # of where the result came from into a claim we ran it ourselves.
            build_env=artifact.build_env,
            build_duration_ms=artifact.build_duration_ms,
            # Persisted, not just logged. The lineage view reads the local
            # store, so without this the author column stays blank on exactly
            # the steps someone else produced — the case it exists for.
            principal=artifact.principal,
        )
        total_bytes += len(artifact.blob)

    # Every variable of one cell run was computed together, so they share an
    # author; reporting the first is reporting all of them.
    principal = pulled[0][1].principal
    build_env = pulled[0][1].build_env
    # One cell run produced every variable together, so its cost is the cost of
    # the run — not the sum over variables, which would multiply it by the
    # number of things the cell happened to define.
    saved_ms = max(artifact.build_duration_ms for _, artifact in pulled)
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

    Reads back what was just written to the *local* store rather than the
    harness's temp files: those artifacts are the ones the pull will have to
    reproduce byte for byte, so publishing anything else would let the two
    halves disagree without either being obviously wrong. It also means the
    per-variable provenance key comes from the stored artifact rather than
    being recomputed here, so the two sides cannot drift.

    Unlike the pull this is not all-or-nothing. A pull that is short one
    variable is a hit that will fail validation, so it must abort; a publish
    that is short one variable just means the next person pulls nothing and
    recomputes — the same outcome as not publishing at all, and no worse for
    having published the others.

    Returns how many variables landed. Never raises: the cell has already
    succeeded, and a shared-cache problem must not retroactively fail it.
    """
    published = 0
    for var in sorted(consumed_vars):
        artifact_id = artifact_mgr.cell_artifact_id(cell_id, var, variant=variant)
        stored = artifact_mgr.artifact_store.get_latest_version(artifact_id)
        if stored is None or stored.state not in ("ready", "superseded"):
            continue
        blob = artifact_mgr.artifact_store.blob_store.read_blob(artifact_id, stored.version)
        if blob is None:
            continue
        if await store.publish(
            stored.provenance_hash,
            blob,
            content_type=_spec_param(stored, "content_type"),
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


def _spec_param(artifact, key: str) -> str:
    """One value out of the stored transform spec's params.

    Carries ``content_type`` (how the blob was serialized) and ``build_env``
    (which interpreter on which machine produced it). Empty rather than a
    guess: the puller needs the first to decode at all, and a plausible default
    that is wrong for a pickle is worse than an absent one.
    """
    if not artifact.transform_spec:
        return ""
    try:
        spec = json.loads(artifact.transform_spec)
    except ValueError:
        return ""
    if not isinstance(spec, dict):
        return ""
    params = spec.get("params")
    if not isinstance(params, dict):
        return ""
    return str(params.get(key) or "")


def _spec_int(artifact, key: str) -> int:
    """The same, for a param that is a whole number of milliseconds."""
    try:
        return int(_spec_param(artifact, key))
    except ValueError:
        return 0
