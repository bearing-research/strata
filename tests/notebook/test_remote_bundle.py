"""Tests for notebook remote bundle transport helpers."""

from __future__ import annotations

import json

from strata.notebook.remote_bundle import (
    SCHEMA_VERSION,
    pack_notebook_output_bundle,
    unpack_notebook_output_bundle,
)


def test_remote_bundle_round_trip_success(tmp_path):
    """A successful harness result should survive pack/unpack losslessly."""
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
    """Failure manifests should preserve stderr, traceback, and schema version."""
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
    manifest = json.loads((unpacked_dir / "result.json").read_text(encoding="utf-8"))
    assert manifest["success"] is False

    # Bundle schema version is the transport contract, not the harness result contract.
    import tarfile

    with tarfile.open(bundle_path, "r") as tar:
        extracted = tar.extractfile("manifest.json")
        assert extracted is not None
        bundle_manifest = json.loads(extracted.read().decode("utf-8"))
    assert bundle_manifest["schema_version"] == SCHEMA_VERSION
