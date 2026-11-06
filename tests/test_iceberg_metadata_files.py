"""Iceberg metadata files are read from the store once, then from Strata's cache."""

from __future__ import annotations

import io
import os
import pickle
import shutil
from typing import ClassVar

import pytest
from pyiceberg.io import FileIO, InputFile, InputStream, OutputFile, OutputStream

from strata.iceberg import metadata_files
from strata.iceberg.metadata_files import CachedFileIO


class _Store:
    """Bytes by location, unreachable while ``down``."""

    files: ClassVar[dict[str, bytes]] = {}
    down: ClassVar[bool] = False
    reads: ClassVar[list[str]] = []


class _StoreInput(InputFile):
    def __len__(self) -> int:
        return len(self._read())

    def exists(self) -> bool:
        return self.location in _Store.files

    def open(self, seekable: bool = True) -> InputStream:
        return io.BytesIO(self._read())  # type: ignore[return-value]

    def _read(self) -> bytes:
        if _Store.down:
            raise ConnectionError(f"store down: {self.location}")
        _Store.reads.append(self.location)
        return _Store.files[self.location]


class _StoreOutput(OutputFile):
    def __len__(self) -> int:
        return len(_Store.files[self.location])

    def exists(self) -> bool:
        return self.location in _Store.files

    def to_input_file(self) -> InputFile:
        return _StoreInput(self.location)

    def create(self, overwrite: bool = False) -> OutputStream:
        location = self.location

        class _Writer(io.BytesIO):
            def close(self) -> None:
                _Store.files[location] = self.getvalue()
                super().close()

        return _Writer()  # type: ignore[return-value]


class StoreFileIO(FileIO):
    """The catalog's own FileIO, named by its py-io-impl."""

    def new_input(self, location: str) -> InputFile:
        return _StoreInput(location)

    def new_output(self, location: str) -> OutputFile:
        return _StoreOutput(location)

    def delete(self, location: str | InputFile | OutputFile) -> None:
        path = location if isinstance(location, str) else location.location
        del _Store.files[path]


METADATA = "s3://lake/t/metadata/00001-5f2c.metadata.json"
MANIFEST = "s3://lake/t/metadata/snap-1-0-ab.avro"


@pytest.fixture(autouse=True)
def _store():
    _Store.files = {METADATA: b'{"format-version": 2}', MANIFEST: b"Obj\x01manifest"}
    _Store.down = False
    _Store.reads = []
    metadata_files._caches.clear()
    yield
    metadata_files._caches.clear()


def _io(tmp_path, size_bytes: int = 1 << 20, **properties: str) -> CachedFileIO:
    props = metadata_files.catalog_properties(
        {"py-io-impl": f"{__name__}.StoreFileIO", **properties},
        tmp_path / "iceberg-metadata",
        size_bytes,
    )
    return CachedFileIO(props)


def _read(file_io: FileIO, location: str) -> bytes:
    with file_io.new_input(location).open() as stream:
        return stream.read()


def test_metadata_read_once_is_read_while_the_store_is_down(tmp_path):
    file_io = _io(tmp_path)
    assert _read(file_io, METADATA) == _Store.files[METADATA]
    assert _read(file_io, MANIFEST) == _Store.files[MANIFEST]

    _Store.down = True
    # A new FileIO, as pyiceberg builds one per table load.
    again = _io(tmp_path)
    assert _read(again, METADATA) == b'{"format-version": 2}'
    assert _read(again, MANIFEST) == b"Obj\x01manifest"
    assert len(again.new_input(METADATA)) == len(b'{"format-version": 2}')
    assert again.new_input(METADATA).exists()


def test_a_hit_does_not_read_the_store(tmp_path):
    file_io = _io(tmp_path)
    _read(file_io, METADATA)
    _read(file_io, METADATA)
    assert _Store.reads == [METADATA]


@pytest.mark.parametrize(
    "location",
    [
        "s3://lake/t/data/00000-0-ab.parquet",
        "s3://lake/t/metadata/version-hint.text",
        "/warehouse/t/metadata/00001-5f2c.metadata.json",
        "file:///warehouse/t/metadata/00001-5f2c.metadata.json",
    ],
)
def test_data_files_changing_files_and_local_files_are_not_cached(tmp_path, location):
    _Store.files[location] = b"bytes"
    file_io = _io(tmp_path)
    _read(file_io, location)
    _Store.down = True
    with pytest.raises(ConnectionError):
        _read(file_io, location)


def test_a_write_does_not_fill_the_cache(tmp_path):
    location = "s3://lake/t/metadata/00002-9a1d.metadata.json"
    file_io = _io(tmp_path)
    with file_io.new_output(location).create() as stream:
        stream.write(b"written")
    assert _Store.files[location] == b"written"
    _Store.down = True
    with pytest.raises(ConnectionError):
        _read(file_io, location)


def test_one_location_on_two_stores_is_two_entries(tmp_path):
    _read(_io(tmp_path, **{"s3.endpoint": "http://minio:9000"}), METADATA)
    _Store.files[METADATA] = b"another store's file"
    assert _read(_io(tmp_path, **{"s3.endpoint": "http://r2:443"}), METADATA) == (
        b"another store's file"
    )


def test_the_least_recently_read_is_evicted_past_the_size(tmp_path):
    first, second, third = (f"s3://lake/t/metadata/0000{i}-a.metadata.json" for i in (1, 2, 3))
    for location in (first, second, third):
        _Store.files[location] = b"x" * 40
    file_io = _io(tmp_path, size_bytes=100)
    _read(file_io, first)
    _read(file_io, second)
    directory = tmp_path / "iceberg-metadata"
    for age, path in enumerate(sorted(directory.iterdir(), key=lambda p: p.stat().st_mtime)):
        os.utime(path, (age, age))
    _read(file_io, first)  # now the most recently read

    _read(file_io, third)  # 120 bytes: past the size

    _Store.down = True
    assert _read(file_io, first) == b"x" * 40
    assert _read(file_io, third) == b"x" * 40
    with pytest.raises(ConnectionError):
        _read(file_io, second)


def test_a_cleared_cache_directory_fills_again(tmp_path):
    file_io = _io(tmp_path)
    shutil.rmtree(tmp_path / "iceberg-metadata")
    _read(file_io, METADATA)
    _Store.down = True
    assert _read(file_io, METADATA) == _Store.files[METADATA]


def test_a_full_disk_still_returns_what_the_store_read(tmp_path, monkeypatch):
    def full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(metadata_files, "_write_durably", full)
    assert _read(_io(tmp_path), METADATA) == _Store.files[METADATA]


def test_it_pickles(tmp_path):
    file_io = _io(tmp_path)
    _read(file_io, METADATA)
    _Store.down = True
    assert _read(pickle.loads(pickle.dumps(file_io)), METADATA) == _Store.files[METADATA]
