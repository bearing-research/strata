"""Tests for notebook remote bundle transport helpers."""

from __future__ import annotations

import json

from strata.notebook.remote_bundle import (
    SCHEMA_VERSION,
    pack_notebook_output_bundle,
    unpack_notebook_output_bundle,
)


def test_remote_bundle_round_trip_success(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "x.json").write_text('{"value": 1}', encoding="utf-8")
    result = {
        "success": True,
        "variables": {
            "x": {
                "content_type": "json/object",
                "file": "x.json",
                "preview": {"value": 1},
            }
        },
        "stdout": "hello\n",
        "stderr": "",
        "mutation_warnings": [],
    }

    bundle_path = tmp_path / "bundle.tar"
    pack_notebook_output_bundle(bundle_path, result, output_dir)

    unpacked_dir = tmp_path / "unpacked"
    unpacked = unpack_notebook_output_bundle(bundle_path, unpacked_dir)

    assert unpacked["success"] is True
    assert unpacked["stdout"] == "hello\n"
    assert unpacked["variables"]["x"]["file"] == "x.json"
    assert json.loads((unpacked_dir / "x.json").read_text(encoding="utf-8")) == {"value": 1}


def test_remote_bundle_round_trip_failure(tmp_path):
    """Failure manifests preserve stderr, traceback and schema version."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    result = {
        "success": False,
        "variables": {},
        "stdout": "",
        "stderr": "boom\n",
        "mutation_warnings": [],
        "error": "boom",
        "traceback": "Traceback...",
    }

    bundle_path = tmp_path / "bundle.tar"
    pack_notebook_output_bundle(bundle_path, result, output_dir)

    unpacked_dir = tmp_path / "unpacked"
    unpacked = unpack_notebook_output_bundle(bundle_path, unpacked_dir)

    assert unpacked["success"] is False
    assert unpacked["error"] == "boom"
    assert unpacked["traceback"] == "Traceback..."
    manifest = json.loads((unpacked_dir / "harness-result.json").read_text(encoding="utf-8"))
    assert manifest["success"] is False

    # Bundle schema version is the transport contract, not the harness result contract.
    import tarfile

    with tarfile.open(bundle_path, "r") as tar:
        extracted = tar.extractfile("manifest.json")
        assert extracted is not None
        bundle_manifest = json.loads(extracted.read().decode("utf-8"))
    assert bundle_manifest["schema_version"] == SCHEMA_VERSION


def test_read_member_rejects_oversized_member(tmp_path, monkeypatch):
    """A bundle declaring an enormous member size is refused rather than OOM the unpacker."""
    import io
    import tarfile

    from strata.notebook.remote_bundle import _read_member

    # Set a tiny cap, then build a tar with a member exceeding it.
    monkeypatch.setenv("STRATA_NOTEBOOK_MAX_BUNDLE_MEMBER_BYTES", "16")

    bundle_path = tmp_path / "oversize.tar"
    payload = b"x" * 64
    with tarfile.open(bundle_path, "w") as tar:
        info = tarfile.TarInfo(name="big.bin")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    import pytest

    with tarfile.open(bundle_path, "r") as tar:
        with pytest.raises(ValueError, match="exceeds .*-byte cap"):
            _read_member(tar, "big.bin")


def test_the_workers_build_environment_survives_the_bundle(tmp_path):
    """The bundle carries the worker's identity; the receiver must not substitute its own.

    Otherwise every remotely computed artifact claims no platform.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "x.json").write_text('{"value": 1}', encoding="utf-8")
    worker_env = "cpython-3.12-linux-x86_64"
    result = {
        "success": True,
        "variables": {"x": {"content_type": "json/object", "file": "x.json"}},
        "stdout": "",
        "stderr": "",
        "mutation_warnings": [],
        "build_env": worker_env,
    }

    bundle_path = tmp_path / "bundle.tar"
    pack_notebook_output_bundle(bundle_path, result, output_dir)

    unpacked_dir = tmp_path / "unpacked"
    unpacked_dir.mkdir()
    unpacked = unpack_notebook_output_bundle(bundle_path, unpacked_dir)

    assert unpacked["build_env"] == worker_env


def test_a_bundle_from_a_worker_that_reports_no_platform_is_still_valid(tmp_path):
    """Workers without the field report an empty platform rather than fail to unpack."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "x.json").write_text('{"value": 1}', encoding="utf-8")
    result = {
        "success": True,
        "variables": {"x": {"content_type": "json/object", "file": "x.json"}},
        "stdout": "",
        "stderr": "",
        "mutation_warnings": [],
    }

    bundle_path = tmp_path / "bundle.tar"
    pack_notebook_output_bundle(bundle_path, result, output_dir)

    unpacked_dir = tmp_path / "unpacked"
    unpacked_dir.mkdir()
    assert unpack_notebook_output_bundle(bundle_path, unpacked_dir)["build_env"] == ""


def test_every_display_survives_the_round_trip(tmp_path):
    """Every display arrives in order, file-backed or inline, not only the last one (``_``)."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    for i in range(3):
        (output_dir / f"display_{i}.png").write_bytes(f"png-{i}".encode())
    displays = [
        {"content_type": "image/png", "file": "display_0.png"},
        {"content_type": "text/markdown", "inline": "# heading"},
        {"content_type": "image/png", "file": "display_1.png"},
        {"error": "could not serialize", "type": "Widget"},
        {"content_type": "image/png", "file": "display_2.png"},
    ]
    result = {
        "success": True,
        "variables": {"_": dict(displays[-1])},
        "displays": displays,
        "stdout": "",
        "stderr": "",
        "mutation_warnings": [],
    }

    bundle_path = tmp_path / "bundle.tar"
    pack_notebook_output_bundle(bundle_path, result, output_dir)
    unpacked_dir = tmp_path / "unpacked"
    unpacked = unpack_notebook_output_bundle(bundle_path, unpacked_dir)

    assert unpacked["displays"] == displays
    for i in range(3):
        assert (unpacked_dir / f"display_{i}.png").read_bytes() == f"png-{i}".encode()


def test_a_bundle_from_an_older_worker_has_no_displays(tmp_path):
    """With no display list from the worker, the executor falls back to ``_``."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    result = {"success": True, "variables": {}, "stdout": "", "stderr": ""}
    bundle_path = tmp_path / "bundle.tar"
    pack_notebook_output_bundle(bundle_path, result, output_dir)

    assert unpack_notebook_output_bundle(bundle_path, tmp_path / "u")["displays"] == []
