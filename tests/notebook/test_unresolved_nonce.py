"""Every unresolved-input component takes its random part from ``nonce.unresolved_nonce``.

Each site keeps the shape it had before the helper: 64 hex digits, or 32 for
``@dataset`` (a uuid4 hex). The cellgraph trace recorder replaces the helper with a
counter, so a site that bypasses it would put a random string in a recorded trace.
"""

from __future__ import annotations

import re

import pytest

from strata.notebook import datasets, mounts, nonce, tables
from strata.notebook.credentials import CredentialError
from strata.notebook.fetch import FetchCache
from strata.notebook.models import DatasetSpec, FetchSpec, MountSpec

_HEX64 = "[0-9a-f]{64}"
_FIXED = "f" * 64


def _failing_filesystem(error: Exception):
    def _raise(*_args, **_kwargs):
        raise error

    return _raise


class _NoCredential:
    def storage_options(self, mount):
        raise CredentialError("not configured")


def _table() -> str:
    return tables.random_fingerprint()


def _listing_without_driver(monkeypatch) -> str:
    monkeypatch.setattr(mounts, "_mount_filesystem", _failing_filesystem(ImportError("s3fs")))
    return mounts.MountFingerprinter.remote_listing_sync("s3", "bucket/p")[0]


def _listing_that_failed(monkeypatch) -> str:
    monkeypatch.setattr(mounts, "_mount_filesystem", _failing_filesystem(OSError("refused")))
    return mounts.MountFingerprinter.remote_listing_sync("s3", "bucket/p")[0]


def _mount_without_credential(monkeypatch) -> str:
    mount = MountSpec(name="raw", uri="s3://bucket/p")
    component, _weight, _approval = mounts.weigh_mount_sync(_NoCredential(), mount, None)
    assert component is not None
    return component


def _fetch(tmp_path) -> str:
    return FetchCache(tmp_path).fingerprint(FetchSpec(name="zones", url="https://example.com/z"))


def _dataset() -> str:
    return datasets.unresolved_fingerprint(DatasetSpec(name="model", dataset="taxi/model"))


# (site, its component with the nonce as ``{}``, the nonce's width)
_SITES = [
    ("tables", lambda mp, tp: _table(), "{}", 64),
    ("mount_no_driver", lambda mp, tp: _listing_without_driver(mp), "{}", 64),
    ("mount_listing_failed", lambda mp, tp: _listing_that_failed(mp), "{}", 64),
    ("mount_no_credential", lambda mp, tp: _mount_without_credential(mp), "raw:unresolved:{}", 64),
    ("fetch", lambda mp, tp: _fetch(tp), "zones:fetch:unresolved:{}", 64),
    ("dataset", lambda mp, tp: _dataset(), "model:dataset:taxi/model:unresolved:{}", 32),
]


@pytest.mark.parametrize(("site", "make", "shape", "width"), _SITES, ids=[s[0] for s in _SITES])
def test_the_component_keeps_its_shape(site, make, shape, width, monkeypatch, tmp_path):
    pattern = re.escape(shape).replace(r"\{\}", f"[0-9a-f]{{{width}}}")
    assert re.fullmatch(pattern, make(monkeypatch, tmp_path))


@pytest.mark.parametrize(("site", "make", "shape", "width"), _SITES, ids=[s[0] for s in _SITES])
def test_the_nonce_comes_from_the_helper(site, make, shape, width, monkeypatch, tmp_path):
    monkeypatch.setattr(nonce, "unresolved_nonce", lambda: _FIXED)
    assert make(monkeypatch, tmp_path) == shape.format(_FIXED[:width])


def test_two_calls_differ():
    assert re.fullmatch(_HEX64, nonce.unresolved_nonce())
    assert nonce.unresolved_nonce() != nonce.unresolved_nonce()
