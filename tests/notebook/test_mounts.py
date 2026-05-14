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
    """Small fake fsspec filesystem for mount resolver tests."""

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
    """Pinned local mounts should use the pin value, not the local file state."""
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
    """Remote RO mounts should mirror nested files, not only the top level."""
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
    """A failed remote mirror should be rebuilt on the next attempt."""
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


@pytest.mark.asyncio
async def test_sync_back_raises_on_remote_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RW sync failures should surface as errors, not just logs."""
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
    """RW staging should not preserve files deleted from the remote mount."""
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
    """A remote returning '..' in a file name must be rejected, not materialized
    above the mount root — the harness only sees pathlib.Path objects rooted
    under local_mirror and a silent escape would widen what the cell can read.
    """
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
    """fsspec's recursive=True is opaque about local paths; the post-fetch
    walk must catch symlinks that point outside the staging dir.
    """
    sentinel = tmp_path / "secret.txt"
    sentinel.write_text("not-for-cell", encoding="utf-8")

    class _SymlinkFakeFS(_FakeRemoteFS):
        def get(self, uri: str, local_path: str, recursive: bool = False) -> None:
            staging = Path(local_path)
            staging.mkdir(parents=True, exist_ok=True)
            (staging / "leak").symlink_to(sentinel)

        def exists(self, uri: str) -> bool:  # noqa: ARG002
            return True

    _install_fake_fsspec(monkeypatch, _SymlinkFakeFS({}))

    resolver = MountResolver(cache_dir=tmp_path / "cache")
    mount = MountSpec(name="m", uri="s3://bucket/output", mode=MountMode.READ_WRITE)

    with pytest.raises(RuntimeError, match="path-traversal|outside the mount root"):
        await resolver.prepare_mounts([mount])


def test_remote_fingerprint_without_fsspec_is_non_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When fsspec is missing, fingerprint_remote_sync used to return a
    deterministic hash, causing the cell to silently cache-hit across
    environments with completely different remote content. It must now
    return a unique-per-call value so the cell is treated as uncacheable.
    """
    from strata.notebook.mounts import MountFingerprinter

    monkeypatch.setitem(sys.modules, "fsspec", None)

    fp1 = MountFingerprinter.fingerprint_remote_sync("s3", "bucket/prefix")
    fp2 = MountFingerprinter.fingerprint_remote_sync("s3", "bucket/prefix")

    assert fp1 != fp2, "fingerprint must be unique per call when fsspec is missing"
