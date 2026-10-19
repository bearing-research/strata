"""Tests for notebook mount resolution and fingerprinting."""

from __future__ import annotations

import hashlib
import io
import sys
import types
from pathlib import Path

import pytest

from strata.notebook.models import MountMode, MountSpec
from strata.notebook.mounts import MountResolver


class _FakeRemoteFS:
    """Small fake fsspec filesystem."""

    def __init__(
        self,
        files: dict[str, dict[str, object]],
        *,
        fail_put: bool = False,
    ) -> None:
        self._files = files
        self._fail_put = fail_put

    def _strip(self, uri: str) -> str:
        return uri.split("://", 1)[-1]

    def isfile(self, uri: str) -> bool:
        return self._strip(uri) in self._files

    def info(self, uri: str) -> dict[str, object]:
        return dict(self._files[self._strip(uri)])

    def find(
        self,
        uri: str,
        *,
        withdirs: bool = False,
        detail: bool = True,
    ) -> dict[str, dict[str, object]]:
        del withdirs, detail
        prefix = self._strip(uri).rstrip("/")
        return {
            name: dict(info)
            for name, info in self._files.items()
            if name == prefix or name.startswith(f"{prefix}/")
        }

    def get(self, uri: str, local_path: str, recursive: bool = False) -> None:
        key = self._strip(uri)
        if key in self._files:
            Path(local_path).write_bytes(self._files[key]["content"])  # type: ignore[arg-type]
            return

        if not recursive:
            raise FileNotFoundError(uri)

        prefix = key.rstrip("/")
        for name, info in self._files.items():
            if not name.startswith(f"{prefix}/"):
                continue
            rel = name[len(prefix) + 1 :]
            target = Path(local_path) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(info["content"])  # type: ignore[arg-type]

    def exists(self, uri: str) -> bool:
        key = self._strip(uri).rstrip("/")
        return any(name == key or name.startswith(f"{key}/") for name in self._files)

    def open(self, uri: str, mode: str = "rb") -> io.BytesIO:
        """Writing only, as sync-back uploads: the bytes land in the fake bucket on close."""
        assert mode == "wb"
        if self._fail_put:
            raise RuntimeError("boom")
        files, key = self._files, self._strip(uri)

        class _Upload(io.BytesIO):
            def close(self) -> None:
                if not self.closed:
                    data = self.getvalue()
                    files[key] = {"name": key, "size": len(data), "content": data}
                super().close()

        return _Upload()


def _install_fake_fsspec(
    monkeypatch: pytest.MonkeyPatch,
    fs: _FakeRemoteFS,
) -> None:
    fake_module = types.SimpleNamespace(filesystem=lambda protocol, **kwargs: fs)
    monkeypatch.setitem(sys.modules, "fsspec", fake_module)


@pytest.mark.asyncio
async def test_local_mount_pin_controls_fingerprint(tmp_path: Path) -> None:
    """Pinned local mounts use the pin value, not the local file state."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "table.parquet").write_text("old", encoding="utf-8")

    resolver = MountResolver(cache_dir=tmp_path / "cache")
    mount = MountSpec(
        name="raw_data",
        uri=f"file://{data_dir}",
        mode=MountMode.READ_ONLY,
        pin="snapshot-123",
    )

    resolved = await resolver.prepare_mounts([mount])

    assert resolved["raw_data"].fingerprint == hashlib.sha256(b"pin:snapshot-123").hexdigest()


@pytest.mark.asyncio
async def test_remote_ro_mount_materializes_nested_files_recursively(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote RO mounts mirror nested files, not only the top level."""
    fs = _FakeRemoteFS(
        {
            "bucket/prefix/a.txt": {
                "name": "bucket/prefix/a.txt",
                "size": 1,
                "etag": "etag-a",
                "mtime": "1",
                "content": b"a",
            },
            "bucket/prefix/nested/b.txt": {
                "name": "bucket/prefix/nested/b.txt",
                "size": 1,
                "etag": "etag-b",
                "mtime": "2",
                "content": b"b",
            },
        }
    )
    _install_fake_fsspec(monkeypatch, fs)

    resolver = MountResolver(cache_dir=tmp_path / "cache")
    mount = MountSpec(
        name="raw_data",
        uri="s3://bucket/prefix",
        mode=MountMode.READ_ONLY,
    )

    resolved = await resolver.prepare_mounts([mount])
    local_path = resolved["raw_data"].local_path

    assert (local_path / "a.txt").read_bytes() == b"a"
    assert (local_path / "nested" / "b.txt").read_bytes() == b"b"
    assert (
        resolved["raw_data"].fingerprint
        == hashlib.sha256(
            "\n".join(
                sorted(
                    [
                        "bucket/prefix/a.txt:1:etag-a:1",
                        "bucket/prefix/nested/b.txt:1:etag-b:2",
                    ]
                )
            ).encode()
        ).hexdigest()
    )


@pytest.mark.asyncio
async def test_remote_ro_mount_retries_after_partial_materialization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed remote mirror is rebuilt on the next attempt."""
    fs = _FakeRemoteFS(
        {
            "bucket/prefix/a.txt": {
                "name": "bucket/prefix/a.txt",
                "size": 1,
                "etag": "etag-a",
                "mtime": "1",
                "content": b"a",
            },
            "bucket/prefix/nested/b.txt": {
                "name": "bucket/prefix/nested/b.txt",
                "size": 1,
                "etag": "etag-b",
                "mtime": "2",
                "content": b"b",
            },
        }
    )
    _install_fake_fsspec(monkeypatch, fs)

    resolver = MountResolver(cache_dir=tmp_path / "cache")
    mount = MountSpec(
        name="raw_data",
        uri="s3://bucket/prefix",
        mode=MountMode.READ_ONLY,
    )

    original_get = fs.get
    call_count = 0

    def flaky_get(uri: str, local_path: str, recursive: bool = False) -> None:
        nonlocal call_count
        call_count += 1
        original_get(uri, local_path, recursive=recursive)
        if call_count == 2:
            raise RuntimeError("network drop")

    monkeypatch.setattr(fs, "get", flaky_get)

    with pytest.raises(RuntimeError, match="Failed to materialize remote mount"):
        await resolver.prepare_mounts([mount])

    monkeypatch.setattr(fs, "get", original_get)
    resolved = await resolver.prepare_mounts([mount])

    assert (resolved["raw_data"].local_path / "a.txt").read_bytes() == b"a"
    assert (resolved["raw_data"].local_path / "nested" / "b.txt").read_bytes() == b"b"


@pytest.mark.parametrize("dirname", ["My Notebooks", "café"])
async def test_local_file_uri_mount_with_percent_encoded_path(tmp_path: Path, dirname) -> None:
    """``Path.as_uri()`` percent-encodes spaces and non-ASCII; the mount must decode it."""
    data = tmp_path / dirname / "zones.csv"
    data.parent.mkdir()
    data.write_text("zone\n1\n")
    mount = MountSpec(name="zones", uri=data.as_uri(), mode=MountMode.READ_ONLY)

    resolved = await MountResolver(cache_dir=tmp_path / "cache").prepare_mounts([mount])

    assert resolved["zones"].local_path == data
    assert resolved["zones"].local_path.read_text() == "zone\n1\n"


def test_resolver_credentials_default_to_empty_dict(tmp_path: Path) -> None:
    resolver = MountResolver(cache_dir=tmp_path / "cache")
    assert resolver.credentials == {}


@pytest.mark.asyncio
async def test_resolver_credentials_reach_fsspec_filesystem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-scheme credentials are spread into ``fsspec.filesystem`` kwargs."""
    fs = _FakeRemoteFS(
        {
            "bucket/prefix/a.txt": {
                "name": "bucket/prefix/a.txt",
                "size": 1,
                "etag": "etag-a",
                "mtime": "1",
                "content": b"a",
            },
        }
    )
    captured: list[tuple[str, dict[str, object]]] = []

    def filesystem(protocol: str, **kwargs: object) -> _FakeRemoteFS:
        captured.append((protocol, kwargs))
        return fs

    monkeypatch.setitem(sys.modules, "fsspec", types.SimpleNamespace(filesystem=filesystem))

    resolver = MountResolver(
        cache_dir=tmp_path / "cache",
        credentials={"s3": {"endpoint_url": "http://minio:9000", "anon": True}},
    )
    await resolver.prepare_mounts(
        [MountSpec(name="raw", uri="s3://bucket/prefix", mode=MountMode.READ_ONLY)]
    )

    assert captured, "fsspec.filesystem was never called"
    protocol, kwargs = captured[0]
    assert protocol == "s3"
    assert kwargs == {"endpoint_url": "http://minio:9000", "anon": True}


@pytest.mark.asyncio
async def test_mount_options_override_credentials_on_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-mount ``options`` win over per-scheme credentials on key collision."""
    fs = _FakeRemoteFS(
        {
            "bucket/prefix/a.txt": {
                "name": "bucket/prefix/a.txt",
                "size": 1,
                "etag": "etag-a",
                "mtime": "1",
                "content": b"a",
            },
        }
    )
    captured: list[dict[str, object]] = []

    def filesystem(protocol: str, **kwargs: object) -> _FakeRemoteFS:
        del protocol
        captured.append(kwargs)
        return fs

    monkeypatch.setitem(sys.modules, "fsspec", types.SimpleNamespace(filesystem=filesystem))

    resolver = MountResolver(
        cache_dir=tmp_path / "cache",
        credentials={"s3": {"endpoint_url": "http://default:9000", "key": "scheme-key"}},
    )
    await resolver.prepare_mounts(
        [
            MountSpec(
                name="raw",
                uri="s3://bucket/prefix",
                mode=MountMode.READ_ONLY,
                options={"endpoint_url": "http://mount-specific:9000"},
            )
        ]
    )

    assert captured[0] == {
        "endpoint_url": "http://mount-specific:9000",  # mount.options wins
        "key": "scheme-key",  # credentials passes through where no override
    }


@pytest.mark.asyncio
async def test_sync_back_raises_on_remote_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RW sync failures raise, not just log."""
    fs = _FakeRemoteFS({}, fail_put=True)
    _install_fake_fsspec(monkeypatch, fs)

    resolver = MountResolver(cache_dir=tmp_path / "cache")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "result.txt").write_text("done", encoding="utf-8")

    resolved = await resolver.prepare_mounts(
        [
            MountSpec(
                name="scratch",
                uri="s3://bucket/output",
                mode=MountMode.READ_WRITE,
            )
        ]
    )
    resolved["scratch"].local_path.mkdir(parents=True, exist_ok=True)
    (resolved["scratch"].local_path / "result.txt").write_text("done", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Failed to sync-back RW mount 'scratch'"):
        await resolver.sync_back(resolved)


@pytest.mark.asyncio
async def test_remote_rw_mount_replaces_staging_with_current_remote_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RW staging drops files deleted from the remote."""
    fs = _FakeRemoteFS(
        {
            "bucket/output/old.txt": {
                "name": "bucket/output/old.txt",
                "size": 3,
                "etag": "etag-old",
                "mtime": "1",
                "content": b"old",
            },
        }
    )
    _install_fake_fsspec(monkeypatch, fs)

    resolver = MountResolver(cache_dir=tmp_path / "cache")
    mount = MountSpec(
        name="scratch",
        uri="s3://bucket/output",
        mode=MountMode.READ_WRITE,
    )

    first = await resolver.prepare_mounts([mount])
    assert (first["scratch"].local_path / "old.txt").read_bytes() == b"old"

    fs._files = {}
    second = await resolver.prepare_mounts([mount])

    assert not (second["scratch"].local_path / "old.txt").exists()


@pytest.mark.asyncio
async def test_remote_ro_mount_rejects_path_traversal_in_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remote file name with '..' is rejected rather than written above the mount root."""
    fs = _FakeRemoteFS(
        {
            "bucket/prefix/../escape.txt": {
                "name": "bucket/prefix/../escape.txt",
                "size": 1,
                "etag": "etag-x",
                "mtime": "1",
                "content": b"x",
            },
        }
    )
    _install_fake_fsspec(monkeypatch, fs)

    resolver = MountResolver(cache_dir=tmp_path / "cache")
    mount = MountSpec(name="m", uri="s3://bucket/prefix", mode=MountMode.READ_ONLY)

    with pytest.raises(RuntimeError, match="path-traversal|outside the mount root"):
        await resolver.prepare_mounts([mount])


@pytest.mark.asyncio
async def test_remote_rw_mount_rejects_path_traversal_via_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The post-fetch walk catches symlinks pointing outside staging; fsspec's
    recursive fetch is opaque about local paths.
    """
    sentinel = tmp_path / "secret.txt"
    sentinel.write_text("not-for-cell", encoding="utf-8")

    class _SymlinkFakeFS(_FakeRemoteFS):
        def get(self, uri: str, local_path: str, recursive: bool = False) -> None:
            # The per-file fetch plants a symlink instead of file bytes, modelling a
            # backend/localfs copy that preserves a hostile link.
            target = Path(local_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(sentinel)

        def exists(self, uri: str) -> bool:  # noqa: ARG002
            return True

    _install_fake_fsspec(
        monkeypatch, _SymlinkFakeFS({"bucket/output/leak": {"size": 1, "type": "file"}})
    )

    resolver = MountResolver(cache_dir=tmp_path / "cache")
    mount = MountSpec(name="m", uri="s3://bucket/output", mode=MountMode.READ_WRITE)

    with pytest.raises(RuntimeError, match="path-traversal|outside the mount root"):
        await resolver.prepare_mounts([mount])


def test_remote_fingerprint_without_fsspec_is_non_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without fsspec the remote fingerprint is unique per call, so the cell never cache-hits.

    A deterministic hash would hit across environments with different remote content.
    """
    from strata.notebook.mounts import MountFingerprinter

    monkeypatch.setitem(sys.modules, "fsspec", None)

    fp1 = MountFingerprinter.fingerprint_remote_sync("s3", "bucket/prefix")
    fp2 = MountFingerprinter.fingerprint_remote_sync("s3", "bucket/prefix")

    assert fp1 != fp2, "fingerprint must be unique per call when fsspec is missing"


@pytest.mark.asyncio
async def test_remote_rw_mount_rejects_traversal_name_before_any_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A '..' name is rejected before any bytes land; a scan of staging would never see it."""
    fetched: list[str] = []

    class _TraversalFakeFS(_FakeRemoteFS):
        def get(self, uri: str, local_path: str, recursive: bool = False) -> None:
            fetched.append(uri)
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            Path(local_path).write_text("x", encoding="utf-8")

        def exists(self, uri: str) -> bool:  # noqa: ARG002
            return True

    _install_fake_fsspec(
        monkeypatch,
        _TraversalFakeFS({"bucket/output/../../escape.txt": {"size": 1, "type": "file"}}),
    )

    resolver = MountResolver(cache_dir=tmp_path / "cache")
    mount = MountSpec(name="m", uri="s3://bucket/output", mode=MountMode.READ_WRITE)

    with pytest.raises(RuntimeError, match="path-traversal|outside the mount root"):
        await resolver.prepare_mounts([mount])
    assert fetched == []  # rejected before a single byte was written


class TestGcsMountFilesystem:
    """A non-Google GCS endpoint gets gcsfs's standard filesystem.

    gcsfs 2026.6+ probes the bucket over Google's gRPC Storage Control API, which
    another endpoint cannot answer, so it retried for about two minutes per bucket.
    """

    @staticmethod
    def _class(options: dict, monkeypatch: pytest.MonkeyPatch, emulator: str | None = None):
        from strata.notebook.mounts import _mount_filesystem

        monkeypatch.delenv("STORAGE_EMULATOR_HOST", raising=False)
        if emulator:
            monkeypatch.setenv("STORAGE_EMULATOR_HOST", emulator)
        return type(_mount_filesystem("gcs", {"token": "anon", **options}))

    def test_an_emulator_endpoint_skips_the_grpc_probe(self, monkeypatch):
        from gcsfs.core import GCSFileSystem

        cls = self._class({"endpoint_url": "http://localhost:4443"}, monkeypatch)
        assert cls is GCSFileSystem

    def test_the_emulator_environment_variable_counts_too(self, monkeypatch):
        from gcsfs.core import GCSFileSystem

        assert self._class({}, monkeypatch, emulator="localhost:4443") is GCSFileSystem

    @pytest.mark.parametrize(
        "options",
        [
            {},
            {"endpoint_url": "https://storage.googleapis.com"},
            {"endpoint_url": "https://storage-psc.p.googleapis.com"},
        ],
        ids=["default", "google", "private-google"],
    )
    def test_google_endpoints_keep_gcsfs_default(self, options, monkeypatch):
        import fsspec

        expected = type(fsspec.filesystem("gcs", token="anon", **options))
        assert self._class(options, monkeypatch) is expected


@pytest.mark.parametrize(("scheme", "extra"), [("s3", "s3"), ("gs", "gcs"), ("az", "azure")])
def test_each_remote_mount_scheme_has_an_extra_with_its_filesystem(scheme, extra):
    """Mounts resolve in Strata's own process, so an install extra must carry the library."""
    import re
    import tomllib

    from fsspec.registry import known_implementations

    from strata.notebook.mounts import _scheme_to_fsspec_protocol

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    extras = tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"]
    declared = {re.split(r"[<>=\[ ;]", req, maxsplit=1)[0] for req in extras[extra]}
    implementation = known_implementations[_scheme_to_fsspec_protocol(scheme)]["class"]
    assert implementation.split(".")[0] in declared


class TestWhatACellWrote:
    """``sync_back`` reports the files a run created or changed through a read-write mount."""

    @staticmethod
    def _sha(data: bytes) -> str:
        import hashlib

        return hashlib.sha256(data).hexdigest()

    @pytest.mark.asyncio
    async def test_a_local_mount_reports_new_and_changed_files_only(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / "keep.txt").write_bytes(b"untouched")
        (out / "old.txt").write_bytes(b"before")
        resolver = MountResolver(cache_dir=tmp_path / "cache")
        spec = MountSpec(name="out", uri=out.as_uri(), mode=MountMode.READ_WRITE)
        resolved = await resolver.prepare_mounts([spec])

        (out / "old.txt").write_bytes(b"after, longer")
        (out / "sub").mkdir()
        (out / "sub" / "new.bin").write_bytes(b"\x00\x01")
        (out / "link").symlink_to(out / "keep.txt")

        (write,) = await resolver.sync_back(resolved)

        assert (write.mount, write.uri) == ("out", out.as_uri())
        assert [(f.path, f.size, f.sha256) for f in write.files] == [
            ("old.txt", 13, self._sha(b"after, longer")),
            ("sub/new.bin", 2, self._sha(b"\x00\x01")),
        ]

    @pytest.mark.asyncio
    async def test_a_remote_mount_reports_against_what_was_staged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Staged files the cell left alone are uploaded again but were not written."""
        fs = _FakeRemoteFS(
            {
                "bucket/out/a.txt": {"content": b"a", "size": 1},
                "bucket/out/b.txt": {"content": b"b", "size": 1},
            }
        )
        _install_fake_fsspec(monkeypatch, fs)
        resolver = MountResolver(cache_dir=tmp_path / "cache")
        resolved = await resolver.prepare_mounts(
            [MountSpec(name="out", uri="s3://bucket/out", mode=MountMode.READ_WRITE)]
        )
        staging = resolved["out"].local_path
        (staging / "b.txt").write_bytes(b"bb")
        (staging / "c.txt").write_bytes(b"c")

        (write,) = await resolver.sync_back(resolved)

        assert [f.path for f in write.files] == ["b.txt", "c.txt"]

    @pytest.mark.asyncio
    async def test_a_run_that_writes_nothing_reports_nothing(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / "keep.txt").write_bytes(b"untouched")
        resolver = MountResolver(cache_dir=tmp_path / "cache")
        resolved = await resolver.prepare_mounts(
            [MountSpec(name="out", uri=out.as_uri(), mode=MountMode.READ_WRITE)]
        )

        assert await resolver.sync_back(resolved) == []


class TestSharedMountCache:
    """Read-only remote mounts share one bounded copy per machine (``MountCache``)."""

    @staticmethod
    def _files(prefix: str, contents: dict[str, bytes]) -> dict[str, dict[str, object]]:
        return {
            f"{prefix}/{name}": {
                "name": f"{prefix}/{name}",
                "size": len(data),
                "etag": hashlib.md5(data).hexdigest(),
                "mtime": "1",
                "content": data,
            }
            for name, data in contents.items()
        }

    @staticmethod
    def _count_gets(monkeypatch: pytest.MonkeyPatch, fs: _FakeRemoteFS) -> list[str]:
        fetched: list[str] = []
        original = fs.get

        def get(uri: str, local_path: str, recursive: bool = False) -> None:
            fetched.append(uri)
            original(uri, local_path, recursive=recursive)

        monkeypatch.setattr(fs, "get", get)
        return fetched

    @staticmethod
    def _spec(uri: str = "s3://bucket/data", **kwargs: object) -> MountSpec:
        return MountSpec(name="data", uri=uri, mode=MountMode.READ_ONLY, **kwargs)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_a_changed_prefix_downloads_only_what_changed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from strata.notebook.mount_cache import MountCache

        fs = _FakeRemoteFS(self._files("bucket/data", {"a": b"aa", "b": b"bb", "c": b"cc"}))
        _install_fake_fsspec(monkeypatch, fs)
        fetched = self._count_gets(monkeypatch, fs)
        resolver = MountResolver(cache_dir=tmp_path / "nb", mount_cache=MountCache(tmp_path / "mc"))

        first = (await resolver.prepare_mounts([self._spec()]))["data"].local_path
        assert len(fetched) == 3

        fs._files.update(self._files("bucket/data", {"d": b"dd", "a": b"a2"}))
        fetched.clear()
        second = (await resolver.prepare_mounts([self._spec()]))["data"].local_path

        assert sorted(fetched) == ["s3://bucket/data/a", "s3://bucket/data/d"]
        assert second != first
        assert {p.name: p.read_bytes() for p in second.iterdir()} == {
            "a": b"a2",
            "b": b"bb",
            "c": b"cc",
            "d": b"dd",
        }
        # The earlier view still reads its own version, through the same bytes where unchanged.
        assert (first / "a").read_bytes() == b"aa"
        assert (first / "b").stat().st_ino == (second / "b").stat().st_ino

    @pytest.mark.asyncio
    async def test_two_notebooks_share_one_copy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from strata.notebook.mount_cache import MountCache

        fs = _FakeRemoteFS(self._files("bucket/data", {"a": b"aa", "b": b"bb"}))
        _install_fake_fsspec(monkeypatch, fs)
        fetched = self._count_gets(monkeypatch, fs)
        cache = MountCache(tmp_path / "mc")
        one = MountResolver(cache_dir=tmp_path / "nb1", mount_cache=cache)
        two = MountResolver(cache_dir=tmp_path / "nb2", mount_cache=cache)

        a = (await one.prepare_mounts([self._spec()]))["data"].local_path
        b = (await two.prepare_mounts([self._spec()]))["data"].local_path

        assert len(fetched) == 2
        assert a == b

    @pytest.mark.asyncio
    async def test_other_storage_options_get_their_own_view(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Neither a pinned view nor its objects reach a caller using other storage options.

        A credential that may list a prefix but not read it must not get another notebook's
        bytes by matching a listing.
        """
        from strata.notebook.mount_cache import MountCache

        fs = _FakeRemoteFS(self._files("bucket/data", {"a": b"aa"}))
        _install_fake_fsspec(monkeypatch, fs)
        fetched = self._count_gets(monkeypatch, fs)
        cache = MountCache(tmp_path / "mc")
        listings: list[str] = []
        original_find = fs.find

        def find(uri: str, **kwargs: object) -> dict[str, dict[str, object]]:
            listings.append(uri)
            return original_find(uri, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(fs, "find", find)

        pinned = self._spec(pin="v1", options={"key": "lab-a"})
        a = (await MountResolver(mount_cache=cache).prepare_mounts([pinned]))["data"].local_path
        again = (await MountResolver(mount_cache=cache).prepare_mounts([pinned]))["data"]
        assert again.local_path == a
        assert len(listings) == 1  # a pinned view is reused without asking the store

        other = self._spec(pin="v1", options={"key": "lab-b"})
        b = (await MountResolver(mount_cache=cache).prepare_mounts([other]))["data"].local_path

        assert b != a
        assert len(listings) == 2  # the store was asked, with lab-b's options
        assert len(fetched) == 2  # and the bytes fetched with them, not linked from lab-a's

    @pytest.mark.asyncio
    async def test_another_endpoint_does_not_reuse_objects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from strata.notebook.mount_cache import MountCache

        fs = _FakeRemoteFS(self._files("bucket/data", {"a": b"aa"}))
        _install_fake_fsspec(monkeypatch, fs)
        fetched = self._count_gets(monkeypatch, fs)
        resolver = MountResolver(mount_cache=MountCache(tmp_path / "mc"))

        await resolver.prepare_mounts([self._spec(options={"endpoint_url": "http://one"})])
        await resolver.prepare_mounts([self._spec(options={"endpoint_url": "http://two"})])

        assert len(fetched) == 2

    @pytest.mark.skipif(sys.platform == "win32", reason="objects are made read-only on POSIX")
    @pytest.mark.asyncio
    async def test_a_cell_cannot_change_the_shared_copy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        from strata.notebook.mount_cache import MountCache

        if os.geteuid() == 0:
            pytest.skip("root ignores file modes")
        fs = _FakeRemoteFS(self._files("bucket/data", {"a": b"aa"}))
        _install_fake_fsspec(monkeypatch, fs)
        resolver = MountResolver(mount_cache=MountCache(tmp_path / "mc"))
        view = (await resolver.prepare_mounts([self._spec()]))["data"].local_path

        with pytest.raises(PermissionError):
            (view / "a").write_bytes(b"changed")

    @pytest.mark.skipif(sys.platform == "win32", reason="views are made read-only on POSIX")
    @pytest.mark.asyncio
    async def test_a_cell_cannot_remove_or_add_files_in_the_shared_view(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every notebook reading the prefix sees the view; eviction still removes it."""
        import os
        import time

        from strata.notebook.mount_cache import MountCache

        if os.geteuid() == 0:
            pytest.skip("root ignores file modes")
        files = self._files("bucket/data", {"a": b"aa", "sub/b": b"bb"})
        # No etag or mtime: downloaded into the view itself, not shared from objects/.
        files["bucket/data/c"] = {"name": "bucket/data/c", "size": 2, "content": b"cc"}
        fs = _FakeRemoteFS(files)
        _install_fake_fsspec(monkeypatch, fs)
        cache = MountCache(tmp_path / "mc", max_idle_days=1)
        view = (await MountResolver(mount_cache=cache).prepare_mounts([self._spec()]))[
            "data"
        ].local_path

        for path in (view / "a", view / "sub" / "b", view / "c"):
            with pytest.raises(PermissionError):
                path.unlink()
        with pytest.raises(PermissionError):
            (view / "new").write_bytes(b"new")
        with pytest.raises(PermissionError):
            (view / "c").write_bytes(b"changed")
        assert sorted(p.name for p in view.rglob("*")) == ["a", "b", "c", "sub"]

        cache.sweep(force=True, now=time.time() + 86400 + 60)
        assert not view.parent.exists()
        assert list(cache.views.iterdir()) == []

    @pytest.mark.asyncio
    async def test_a_view_a_run_holds_is_kept_however_long_it_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Its lock is touched when the run starts; a sweep after an hour must still spare it."""
        import gc

        from strata.notebook.mount_cache import MountCache

        fs = _FakeRemoteFS(self._files("bucket/data", {"a": b"x" * 10}))
        _install_fake_fsspec(monkeypatch, fs)
        cache = MountCache(tmp_path / "mc", max_bytes=5)
        resolved = await MountResolver(mount_cache=cache).prepare_mounts([self._spec()])
        view = resolved["data"].local_path
        self._age_view(view, 3 * 3600)

        cache.sweep(force=True)
        assert (view / "a").read_bytes() == b"x" * 10

        del resolved
        gc.collect()
        cache.sweep(force=True)
        assert not view.exists()

    @staticmethod
    def _age_view(view_data: Path, seconds: float) -> None:
        import os
        import time

        lock = view_data.parent / ".lock"
        then = time.time() - seconds
        os.utime(lock, (then, then))

    @pytest.mark.asyncio
    async def test_the_cache_stays_under_its_bound(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Least recently used views go first, and their objects with them."""
        from strata.notebook.mount_cache import MountCache

        fs = _FakeRemoteFS(
            {
                **self._files("bucket/old", {"a": b"x" * 10}),
                **self._files("bucket/mid", {"b": b"y" * 10}),
                **self._files("bucket/new", {"c": b"z" * 10}),
            }
        )
        _install_fake_fsspec(monkeypatch, fs)
        cache = MountCache(tmp_path / "mc", max_bytes=25)
        resolver = MountResolver(mount_cache=cache)

        old = (await resolver.prepare_mounts([self._spec("s3://bucket/old")]))["data"].local_path
        self._age_view(old, 3 * 3600)
        mid = (await resolver.prepare_mounts([self._spec("s3://bucket/mid")]))["data"].local_path
        self._age_view(mid, 2 * 3600)
        new = (await resolver.prepare_mounts([self._spec("s3://bucket/new")]))["data"].local_path

        assert not old.exists()
        assert mid.exists() and new.exists()
        objects = [p for p in (cache.objects).rglob("*") if p.is_file()]
        assert sum(p.stat().st_size for p in objects) == 20

    @pytest.mark.asyncio
    async def test_a_view_used_in_the_last_hour_is_kept_over_the_bound(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A running cell may still be reading it."""
        from strata.notebook.mount_cache import MountCache

        fs = _FakeRemoteFS(
            {
                **self._files("bucket/one", {"a": b"x" * 10}),
                **self._files("bucket/two", {"b": b"y" * 10}),
            }
        )
        _install_fake_fsspec(monkeypatch, fs)
        resolver = MountResolver(mount_cache=MountCache(tmp_path / "mc", max_bytes=5))

        one = (await resolver.prepare_mounts([self._spec("s3://bucket/one")]))["data"].local_path
        two = (await resolver.prepare_mounts([self._spec("s3://bucket/two")]))["data"].local_path

        assert one.exists() and two.exists()

    @pytest.mark.asyncio
    async def test_an_idle_view_is_removed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import time

        from strata.notebook.mount_cache import MountCache

        fs = _FakeRemoteFS(self._files("bucket/data", {"a": b"aa"}))
        _install_fake_fsspec(monkeypatch, fs)
        cache = MountCache(tmp_path / "mc", max_idle_days=1)
        resolver = MountResolver(mount_cache=cache)
        view = (await resolver.prepare_mounts([self._spec()]))["data"].local_path

        cache.sweep(force=True, now=time.time() + 86400 - 60)
        assert view.exists()
        cache.sweep(force=True, now=time.time() + 86400 + 60)
        assert not view.exists()
        assert not [p for p in cache.objects.rglob("*") if p.is_file()]

    def test_a_worker_reads_the_server_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from strata.notebook.remote_executor import _worker_mount_cache

        monkeypatch.setenv("STRATA_NOTEBOOK_MOUNT_CACHE_DIR", str(tmp_path / "mc"))
        monkeypatch.setenv("STRATA_NOTEBOOK_MOUNT_CACHE_MAX_BYTES", "1000")
        monkeypatch.setenv("STRATA_NOTEBOOK_MOUNT_CACHE_MAX_IDLE_DAYS", "not a number")

        cache = _worker_mount_cache()

        assert cache.root == tmp_path / "mc"
        assert cache.max_bytes == 1000
        assert cache.max_idle_seconds == 30 * 86400

    def test_the_cache_defaults_beside_notebook_storage(self, tmp_path: Path) -> None:
        """Where a harness user, which reads the notebooks, can reach it too."""
        from strata.config import StrataConfig
        from strata.notebook.mount_cache import MountCache

        default = StrataConfig(notebook_storage_dir=tmp_path / "notebooks")
        configured = StrataConfig(
            notebook_storage_dir=tmp_path / "notebooks", notebook_mount_cache_dir=tmp_path / "mc"
        )

        assert MountCache.from_config(default).root == tmp_path / "mount_cache"
        assert MountCache.from_config(configured).root == tmp_path / "mc"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
    @pytest.mark.asyncio
    async def test_another_user_cannot_list_the_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Passable, so a harness user reads the view it is handed, but not listable."""
        import stat

        from strata.notebook.mount_cache import MountCache

        fs = _FakeRemoteFS(self._files("bucket/data", {"a": b"aa"}))
        _install_fake_fsspec(monkeypatch, fs)
        cache = MountCache(tmp_path / "mc")
        view = (await MountResolver(mount_cache=cache).prepare_mounts([self._spec()]))[
            "data"
        ].local_path

        shard = next(cache.objects.iterdir())
        for directory in (cache.views, cache.objects, shard):
            assert stat.S_IMODE(directory.stat().st_mode) & 0o044 == 0, directory
        assert stat.S_IMODE((view / "a").stat().st_mode) & 0o222 == 0

    def test_two_names_on_one_path_keep_the_last(self, tmp_path: Path) -> None:
        """A case-insensitive disk folds ``Data.csv`` and ``data.csv``; the view still builds."""
        from strata.notebook.mount_cache import MountCache

        contents = {"remote/A": b"first", "remote/B": b"second"}
        cache = MountCache(tmp_path / "mc")

        view = cache.build_view(
            "k",
            [(Path("same"), "a" * 64, "remote/A"), (Path("same"), "b" * 64, "remote/B")],
            lambda name, dest: dest.write_bytes(contents[name]),
        )

        assert (view / "same").read_bytes() == b"second"

    def test_a_disk_without_hard_links_gets_copies(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from strata.notebook import mount_cache
        from strata.notebook.mount_cache import MountCache

        def no_links(src: object, dst: object) -> None:
            raise PermissionError("operation not permitted")

        monkeypatch.setattr(mount_cache.os, "link", no_links)
        cache = MountCache(tmp_path / "mc")

        view = cache.build_view(
            "k", [(Path("a"), "a" * 64, "remote/a")], lambda name, dest: dest.write_bytes(b"aa")
        )

        assert (view / "a").read_bytes() == b"aa"

    @pytest.mark.asyncio
    async def test_one_sweep_per_prepare(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sweep walks every object; a cell with several mounts pays for one."""
        from strata.notebook.mount_cache import MountCache

        fs = _FakeRemoteFS(
            {**self._files("bucket/one", {"a": b"a"}), **self._files("bucket/two", {"b": b"b"})}
        )
        _install_fake_fsspec(monkeypatch, fs)
        cache = MountCache(tmp_path / "mc")
        sweeps: list[bool] = []
        original = cache.sweep
        monkeypatch.setattr(cache, "sweep", lambda **kw: sweeps.append(True) or original(**kw))

        await MountResolver(mount_cache=cache).prepare_mounts(
            [
                MountSpec(name="one", uri="s3://bucket/one", mode=MountMode.READ_ONLY),
                MountSpec(name="two", uri="s3://bucket/two", mode=MountMode.READ_ONLY),
            ]
        )

        assert len(sweeps) == 1

    @pytest.mark.parametrize("raw", ["inf", "nan", "-5"])
    def test_a_worker_ignores_a_number_it_cannot_use(
        self, raw: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from strata.notebook.remote_executor import _worker_mount_cache

        monkeypatch.setenv("STRATA_NOTEBOOK_MOUNT_CACHE_DIR", str(tmp_path / "mc"))
        monkeypatch.setenv("STRATA_NOTEBOOK_MOUNT_CACHE_MAX_BYTES", raw)

        assert _worker_mount_cache().max_bytes == 20 * 1024**3


class TestAStagingCopyIsTheCellsToWrite:
    """Under the harness user the cell owns its staging copy, so the server follows no link
    in it: neither the record nor the upload may carry a file the cell could not read."""

    @staticmethod
    async def _staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        fs = _FakeRemoteFS({})
        _install_fake_fsspec(monkeypatch, fs)
        resolver = MountResolver(cache_dir=tmp_path / "cache")
        resolved = await resolver.prepare_mounts(
            [MountSpec(name="out", uri="s3://bucket/out", mode=MountMode.READ_WRITE)]
        )
        return fs, resolver, resolved

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
    @pytest.mark.asyncio
    async def test_a_link_is_neither_recorded_nor_uploaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secret = tmp_path / "server-only.txt"
        secret.write_bytes(b"secret")
        (tmp_path / "server-dir").mkdir()
        (tmp_path / "server-dir" / "inner.txt").write_bytes(b"secret too")
        fs, resolver, resolved = await self._staged(tmp_path, monkeypatch)
        staging = resolved["out"].local_path
        (staging / "result.txt").write_bytes(b"mine")
        (staging / "nested").mkdir()
        (staging / "nested" / "deep.txt").write_bytes(b"deep")
        (staging / "leak.txt").symlink_to(secret)
        (staging / "leakdir").symlink_to(tmp_path / "server-dir")

        (write,) = await resolver.sync_back(resolved)

        assert [f.path for f in write.files] == ["nested/deep.txt", "result.txt"]
        assert sorted(fs._files) == ["bucket/out/nested/deep.txt", "bucket/out/result.txt"]
        assert fs._files["bucket/out/result.txt"]["content"] == b"mine"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX hard-link counts")
    @pytest.mark.asyncio
    async def test_a_hard_link_is_neither_recorded_nor_uploaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        secret = tmp_path / "server-only.txt"
        secret.write_bytes(b"secret")
        fs, resolver, resolved = await self._staged(tmp_path, monkeypatch)
        staging = resolved["out"].local_path
        (staging / "result.txt").write_bytes(b"mine")
        os.link(secret, staging / "linked.txt")

        (write,) = await resolver.sync_back(resolved)

        assert [f.path for f in write.files] == ["result.txt"]
        assert sorted(fs._files) == ["bucket/out/result.txt"]

    @pytest.mark.asyncio
    async def test_the_executor_hands_it_to_the_harness_user(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import SimpleNamespace

        from strata.notebook import executor as executor_module
        from strata.notebook.executor import CellExecutor
        from strata.notebook.harness_user import HarnessUser

        _install_fake_fsspec(monkeypatch, _FakeRemoteFS({}))
        user = HarnessUser(name="cells", uid=4242, gid=4242, home="/home/cells")
        calls: list[tuple[str, Path]] = []
        monkeypatch.setattr(executor_module, "resolve_harness_user", lambda: user)
        monkeypatch.setattr(
            executor_module, "hand_over", lambda path, u: calls.append(("hand_over", path))
        )
        monkeypatch.setattr(
            executor_module, "let_through", lambda path, u: calls.append(("let_through", path))
        )
        executor = CellExecutor.__new__(CellExecutor)
        executor._mount_resolver = MountResolver(cache_dir=tmp_path / "cache")
        executor._credential_resolver = lambda: executor._mount_resolver.credential_resolver
        executor._lake_config = lambda: SimpleNamespace()

        resolved = await executor._prepare_mounts(
            [MountSpec(name="out", uri="s3://bucket/out", mode=MountMode.READ_WRITE)]
        )

        staging = resolved["out"].local_path
        assert calls == [
            ("let_through", tmp_path / "cache"),
            ("let_through", staging.parent),
            ("hand_over", staging),
        ]
