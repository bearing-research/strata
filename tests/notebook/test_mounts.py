"""Tests for notebook mount resolution and fingerprinting."""

from __future__ import annotations

import hashlib
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

    def put(self, local_path: str, remote_uri: str, recursive: bool = True) -> None:
        del local_path, remote_uri, recursive
        if self._fail_put:
            raise RuntimeError("boom")


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
