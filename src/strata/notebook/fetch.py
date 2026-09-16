"""Recorded fetches: bytes a cell reads from a URL, as a tracked input (``@fetch``).

A cell that reads a URL with pandas or urllib has the URL in its source hash and
the bytes in nothing, so the same cell can compute different results from the
same provenance. ``@fetch`` makes the bytes an input::

    # @fetch zones https://example.org/taxi_zones.csv
    import pandas as pd
    df = pd.read_csv(zones)

The executor downloads the URL into the notebook's content-addressed cache
(``.strata/fetch/<sha256>/``), injects ``zones`` as a local ``Path`` to the
bytes, and folds ``"<name>:fetch:<url>:<sha256>"`` into the cell's provenance —
so the cell goes stale when the bytes move, the way an ``@table`` does when a
snapshot lands.

``sha256=<digest>`` pins it: the fingerprint is the pin, and bytes that no
longer match fail the cell with both digests. ``refetch`` says when the URL is
checked:

* ``stale`` (default) — a conditional GET (``ETag`` / ``Last-Modified``), at
  most every ``STALE_CHECK_SECONDS`` while staleness is recomputed and always
  right before the cell runs;
* ``never`` — use the bytes already cached; fetch only if there are none;
* ``always`` — download again on every check, ignoring validators.

The recheck interval exists because staleness is recomputed on every source
edit; a request per ``@fetch`` per keystroke would be worse than the problem.
Execution always checks, so what a run records is what the URL served then.

URLs go through the same guard as a worker's manifest URLs: http(s) only, and
no private or link-local address unless the host is named in
``notebook_fetch_allowed_hosts``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx

from strata.notebook.models import FetchSpec
from strata.url_safety import url_safety_problem

STALE_CHECK_SECONDS = 60.0
FETCH_TIMEOUT_SECONDS = 60.0
MAX_FETCH_BYTES = 2 * 1024 * 1024 * 1024
MAX_REDIRECTS = 5

RefetchPolicy = Literal["never", "stale", "always"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


def _safe_filename(name: str) -> str:
    """The URL's own file name, reduced to something that can only name a file.

    Kept at all so a cell that looks at the extension still can; reduced
    because it comes from a URL, where ``..`` is a perfectly good last segment.
    """
    cleaned = _UNSAFE_NAME.sub("_", name).lstrip(".")[:128]
    return cleaned or "data"


class FetchError(RuntimeError):
    """A declared fetch could not produce bytes the cell may use."""


class FetchPinMismatch(FetchError):
    """The URL served bytes that differ from the pinned digest."""

    def __init__(self, spec: FetchSpec, actual: str):
        self.expected = spec.sha256 or ""
        self.actual = actual
        super().__init__(
            f"@fetch {spec.name}: {spec.url} is pinned to sha256={self.expected} "
            f"but served sha256={actual}"
        )


@dataclass(frozen=True)
class FetchedBytes:
    path: Path
    sha256: str

    def fingerprint(self, spec: FetchSpec) -> str:
        return f"{spec.name}:fetch:{spec.url}:{self.sha256}"


class FetchCache:
    """A notebook's fetched bytes, content-addressed, with what each URL served last."""

    def __init__(
        self,
        notebook_dir: Path,
        *,
        allowed_hosts: tuple[str, ...] = (),
        client: httpx.Client | None = None,
        clock=time.time,
    ):
        self.root = Path(notebook_dir) / ".strata" / "fetch"
        self._allowed_hosts = allowed_hosts
        self._client = client
        self._clock = clock

    # -- public ----------------------------------------------------------------

    def resolve(self, spec: FetchSpec, *, max_age: float | None = None) -> FetchedBytes:
        """Bytes for *spec*, checking the URL according to its policy.

        ``max_age=0`` forces the check a ``stale`` fetch makes before a run.

        Raises:
            FetchError: an unsafe URL, a failed download, or (as
                ``FetchPinMismatch``) bytes that differ from the pin.
        """
        if max_age is None:
            max_age = STALE_CHECK_SECONDS
        record = self._index().get(spec.url)
        cached = self._cached(record)

        if spec.sha256 and cached is not None and cached.sha256 == spec.sha256:
            return cached
        if spec.refetch == "never" and cached is not None and not spec.sha256:
            return cached
        if (
            spec.refetch == "stale"
            and cached is not None
            and not spec.sha256
            and self._clock() - float((record or {}).get("checked_at", 0)) < max_age
        ):
            return cached

        fetched = self._download(spec, record if spec.refetch == "stale" else None)
        if spec.sha256 and fetched.sha256 != spec.sha256:
            raise FetchPinMismatch(spec, fetched.sha256)
        return fetched

    def recorded(self, url: str) -> FetchedBytes | None:
        """The bytes last read from *url*, if they are still cached. No network."""
        return self._cached(self._index().get(url))

    def fingerprint(self, spec: FetchSpec, *, max_age: float | None = None) -> str:
        """The provenance component, for staleness. Never raises.

        A pin is its own fingerprint, needing no network. Anything unresolvable
        gets a unique one, the same stance as an unreachable ``@table``: the cell
        shows stale and runs, and the run reports why.
        """
        if spec.sha256:
            return f"{spec.name}:fetch:{spec.url}:{spec.sha256}"
        try:
            return self.resolve(spec, max_age=max_age).fingerprint(spec)
        except FetchError:
            return f"{spec.name}:fetch:unresolved:{hashlib.sha256(os.urandom(32)).hexdigest()}"

    # -- internals ---------------------------------------------------------------

    def _download(self, spec: FetchSpec, record: dict | None) -> FetchedBytes:
        headers: dict[str, str] = {}
        cached = self._cached(record)
        if record and cached is not None:
            if record.get("etag"):
                headers["If-None-Match"] = record["etag"]
            if record.get("last_modified"):
                headers["If-Modified-Since"] = record["last_modified"]

        client = self._client or httpx.Client(timeout=FETCH_TIMEOUT_SECONDS)
        partial = self.root / f".partial-{os.getpid()}-{id(spec)}"
        try:
            url = spec.url
            for _ in range(MAX_REDIRECTS + 1):
                # Every hop through the guard: a client that followed redirects
                # itself would check only the first URL, and a public host that
                # redirects to 169.254.169.254 is the request the guard exists for.
                problem = url_safety_problem(
                    url, f"@fetch {spec.name}", allowed_hosts=self._allowed_hosts
                )
                if problem is not None:
                    raise FetchError(problem)
                with client.stream("GET", url, headers=headers, follow_redirects=False) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("location")
                        if not location:
                            raise FetchError(
                                f"@fetch {spec.name}: {url} redirected without a location"
                            )
                        url = str(response.url.join(location))
                        continue
                    if response.status_code == 304 and cached is not None:
                        self._record(spec.url, cached.sha256, response, record)
                        return cached
                    if response.status_code != 200:
                        raise FetchError(
                            f"@fetch {spec.name}: {spec.url} answered HTTP {response.status_code}"
                        )
                    sha = self._stream_to(partial, response, spec)
                    target = self._blob_path(sha, spec.url)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(partial, target)
                    self._record(spec.url, sha, response, None, filename=target.name)
                    return FetchedBytes(path=target, sha256=sha)
            raise FetchError(
                f"@fetch {spec.name}: {spec.url} redirected more than {MAX_REDIRECTS} times"
            )
        except httpx.HTTPError as exc:
            raise FetchError(f"@fetch {spec.name}: could not download {spec.url}: {exc}") from exc
        finally:
            partial.unlink(missing_ok=True)
            if self._client is None:
                client.close()

    def _stream_to(self, partial: Path, response: httpx.Response, spec: FetchSpec) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        with open(partial, "wb") as out:
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > MAX_FETCH_BYTES:
                    raise FetchError(
                        f"@fetch {spec.name}: {spec.url} exceeds {MAX_FETCH_BYTES} bytes"
                    )
                digest.update(chunk)
                out.write(chunk)
        return digest.hexdigest()

    def _blob_path(self, sha: str, url: str) -> Path:
        return self._contained(sha, _safe_filename(Path(urlparse(url).path).name))

    def _cached(self, record: dict | None) -> FetchedBytes | None:
        if not record:
            return None
        sha = str(record.get("sha256", ""))
        if not _SHA256.fullmatch(sha):
            return None
        try:
            path = self._contained(sha, _safe_filename(str(record.get("filename", ""))))
        except FetchError:
            return None
        return FetchedBytes(path=path, sha256=sha) if path.is_file() else None

    def _contained(self, sha: str, filename: str) -> Path:
        """``<root>/<sha>/<filename>``, refused unless it stays under the root.

        The name comes from a URL, and the index is a file on disk; neither is
        trusted to keep the bytes where they belong.
        """
        if not _SHA256.fullmatch(sha):
            raise FetchError(f"not a sha256 digest: {sha!r}")
        root = self.root.resolve()
        path = (root / sha / filename).resolve()
        if not path.is_relative_to(root / sha):
            raise FetchError(f"fetched file name {filename!r} leaves the fetch cache")
        return path

    def _index_path(self) -> Path:
        return self.root / "index.json"

    def _index(self) -> dict[str, dict]:
        path = self._index_path()
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _record(
        self,
        url: str,
        sha: str,
        response: httpx.Response,
        previous: dict | None,
        *,
        filename: str | None = None,
    ) -> None:
        index = self._index()
        now = self._clock()
        entry = dict(previous or {})
        entry.update(
            {
                "sha256": sha,
                "filename": filename or entry.get("filename"),
                "checked_at": now,
                "etag": response.headers.get("etag") or entry.get("etag"),
                "last_modified": response.headers.get("last-modified")
                or entry.get("last_modified"),
            }
        )
        if filename is not None:
            entry["fetched_at"] = now
        index[url] = entry
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self._index_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self._index_path())
