"""A disk cache of the Iceberg metadata files planning reads from an object store.

Iceberg writes a table's metadata files once, under names it never reuses: each
``metadata.json``, manifest list and manifest. So a copy read once serves every later
read of that location, and a scan of a snapshot already planned plans with no
object-store reads while the catalog answers (#1181).

pyiceberg loads the FileIO named by the ``py-io-impl`` catalog property; Strata names
:class:`CachedFileIO`, which reads through pyiceberg's own choice for each scheme.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from pyiceberg.io import (
    ADLS_ACCOUNT_HOST,
    ADLS_ACCOUNT_NAME,
    ADLS_BLOB_STORAGE_AUTHORITY,
    ADLS_DFS_STORAGE_AUTHORITY,
    GCS_SERVICE_HOST,
    HF_ENDPOINT,
    PY_IO_IMPL,
    S3_ENDPOINT,
    FileIO,
    InputFile,
    InputStream,
    OutputFile,
    load_file_io,
)
from vestigo.file_modes import private_dir

# Catalog properties that configure the cache, kept from the FileIO it reads through.
CACHE_DIR = "strata.metadata-cache.dir"
CACHE_SIZE_BYTES = "strata.metadata-cache.size-bytes"
# A FileIO the catalog's own properties named, read through instead of pyiceberg's choice.
INNER_PY_IO_IMPL = "strata.metadata-cache.py-io-impl"
_OWN = (CACHE_DIR, CACHE_SIZE_BYTES, INNER_PY_IO_IMPL)

# Properties that choose which store a location is on: one bucket name can be on two.
_STORE = (
    S3_ENDPOINT,
    GCS_SERVICE_HOST,
    ADLS_ACCOUNT_NAME,
    ADLS_ACCOUNT_HOST,
    ADLS_BLOB_STORAGE_AUTHORITY,
    ADLS_DFS_STORAGE_AUTHORITY,
    HF_ENDPOINT,
)
# Avro files in a table are its manifest lists and manifests. version-hint.text changes.
_IMMUTABLE_SUFFIXES = (".metadata.json", ".avro")
# A put's temp file older than this was left by a crash; a live put renames it at once,
# and another process sharing the directory may be mid-put.
_STALE_TEMP_SECONDS = 3600

logger = logging.getLogger(__name__)


def catalog_properties(
    properties: dict[str, Any], cache_dir: Path, size_bytes: int
) -> dict[str, Any]:
    """*properties* with :class:`CachedFileIO` as the FileIO, caching under *cache_dir*."""
    props = dict(properties)
    if PY_IO_IMPL in props:
        props[INNER_PY_IO_IMPL] = props[PY_IO_IMPL]
    props[PY_IO_IMPL] = f"{CachedFileIO.__module__}.{CachedFileIO.__qualname__}"
    props[CACHE_DIR] = str(cache_dir)
    props[CACHE_SIZE_BYTES] = str(size_bytes)
    return props


class MetadataFileCache:
    """Files by key under one directory, evicting the least recently read past a size."""

    def __init__(self, directory: Path, max_size_bytes: int) -> None:
        """Count what *directory* holds, removing temp files a crashed put left."""
        self.directory = directory
        self.max_size_bytes = max_size_bytes
        private_dir(directory)
        self._lock = threading.Lock()
        self._size_bytes = 0
        stale_before = time.time() - _STALE_TEMP_SECONDS
        for path in directory.iterdir():
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue  # another process sharing the directory evicted it
            if path.suffix != ".tmp":
                self._size_bytes += stat.st_size
            elif stat.st_mtime < stale_before:
                path.unlink(missing_ok=True)

    def get(self, key: str) -> Path | None:
        """The file cached under *key*, marked as just read, or None."""
        path = self.directory / key
        try:
            os.utime(path)
        except FileNotFoundError:
            return None
        return path

    def put(self, key: str, data: bytes) -> None:
        """Cache *data* under *key*, then evict past the size limit."""
        temp = self.directory / f"{key}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            _write_durably(temp, data)
        except FileNotFoundError:
            # Removed with the rest of the cache directory (POST /v1/cache/clear).
            private_dir(self.directory)
            _write_durably(temp, data)
        os.replace(temp, self.directory / key)
        with self._lock:
            self._size_bytes += len(data)
            if self._size_bytes > self.max_size_bytes:
                self._evict()

    def _evict(self) -> None:
        """Delete the least recently read files down to 80% of the limit, under the lock."""
        files = []
        for path in self.directory.iterdir():
            if path.suffix == ".tmp":
                continue  # a put in flight
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue  # another process sharing the directory evicted it
            files.append((stat.st_mtime, stat.st_size, path))
        files.sort()
        size = sum(file_size for _, file_size, _ in files)
        target = int(self.max_size_bytes * 0.8)
        for _, file_size, path in files:
            if size <= target:
                break
            path.unlink(missing_ok=True)
            size -= file_size
        self._size_bytes = size


def _write_durably(path: Path, data: bytes) -> None:
    """Write *data* to *path*, flushed before a rename: a torn copy would be read forever."""
    with open(path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


_caches: dict[tuple[str, int], MetadataFileCache] = {}
_caches_lock = threading.Lock()


def _cache(directory: str, max_size_bytes: int) -> MetadataFileCache:
    """The process's one cache for *directory*: pyiceberg builds a FileIO per table load."""
    key = (directory, max_size_bytes)
    with _caches_lock:
        cache = _caches.get(key)
        if cache is None:
            cache = _caches[key] = MetadataFileCache(Path(directory), max_size_bytes)
        return cache


class _CachedInputFile(InputFile):
    """A metadata file, read from the cache or else through the store once and kept."""

    def __init__(self, inner: InputFile, cache: MetadataFileCache, key: str) -> None:
        super().__init__(inner.location)
        self._inner = inner
        self._cache = cache
        self._key = key

    def __len__(self) -> int:
        path = self._cache.get(self._key)
        return path.stat().st_size if path is not None else len(self._inner)

    def exists(self) -> bool:
        return self._cache.get(self._key) is not None or self._inner.exists()

    def open(self, seekable: bool = True) -> InputStream:
        path = self._cache.get(self._key)
        # Both are InputStreams; pyiceberg's protocol names an __exit__ parameter differently.
        if path is not None:
            return cast(InputStream, path.open("rb"))
        with self._inner.open(seekable=False) as stream:
            data = stream.read()
        try:
            self._cache.put(self._key, data)
        except OSError as exc:
            # The read succeeded; only the next one goes to the store again.
            logger.warning("Could not cache Iceberg metadata %s: %s", self.location, exc)
        return cast(InputStream, io.BytesIO(data))


class CachedFileIO(FileIO):
    """pyiceberg's FileIO for each scheme, with immutable remote metadata read from a cache.

    Writes, deletes, data files and local paths go straight through.
    """

    def __init__(self, properties: dict[str, str]) -> None:
        super().__init__(properties)
        self._cache = _cache(properties[CACHE_DIR], int(properties[CACHE_SIZE_BYTES]))
        self._inner_properties = {
            k: v for k, v in properties.items() if k not in (*_OWN, PY_IO_IMPL)
        }
        if INNER_PY_IO_IMPL in properties:
            self._inner_properties[PY_IO_IMPL] = properties[INNER_PY_IO_IMPL]
        self._store = "\n".join(properties.get(name, "") for name in _STORE)
        self._inners: dict[str, FileIO] = {}
        self._inners_lock = threading.Lock()

    def __getstate__(self) -> dict[str, str]:
        return dict(self.properties)

    def __setstate__(self, state: dict[str, str]) -> None:
        self.__init__(state)

    def _inner(self, location: str) -> FileIO:
        """pyiceberg's FileIO for *location*'s scheme, built on first use."""
        scheme = urlparse(location).scheme
        with self._inners_lock:
            inner = self._inners.get(scheme)
            if inner is None:
                inner = self._inners[scheme] = load_file_io(self._inner_properties, location)
            return inner

    def new_input(self, location: str) -> InputFile:
        inner = self._inner(location).new_input(location)
        scheme = urlparse(location).scheme
        if scheme in ("", "file") or not location.endswith(_IMMUTABLE_SUFFIXES):
            return inner
        key = hashlib.sha256(f"{self._store}\n{location}".encode()).hexdigest()
        return _CachedInputFile(inner, self._cache, key)

    def new_output(self, location: str) -> OutputFile:
        return self._inner(location).new_output(location)

    def delete(self, location: str | InputFile | OutputFile) -> None:
        path = location if isinstance(location, str) else location.location
        self._inner(path).delete(location)
