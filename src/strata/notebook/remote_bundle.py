"""Bundle helpers for notebook remote-style execution results."""

from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "notebook-output-bundle@v1"

# Per-member size cap. The bundle format expands a member fully into
# memory when read (tar.extractfile + .read()), so an unbounded member
# can OOM the unpacker. Override via STRATA_NOTEBOOK_MAX_BUNDLE_MEMBER_BYTES
# when legitimate outputs exceed this default.
_DEFAULT_MAX_BUNDLE_MEMBER_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB


def _max_bundle_member_bytes() -> int:
    raw = os.environ.get("STRATA_NOTEBOOK_MAX_BUNDLE_MEMBER_BYTES")
    if not raw:
        return _DEFAULT_MAX_BUNDLE_MEMBER_BYTES
    try:
        parsed = int(raw)
    except ValueError:
        return _DEFAULT_MAX_BUNDLE_MEMBER_BYTES
    return parsed if parsed > 0 else _DEFAULT_MAX_BUNDLE_MEMBER_BYTES


def _add_bytes(tar: tarfile.TarFile, arcname: str, content: bytes) -> None:
    """Add in-memory bytes as one tar member."""
    info = tarfile.TarInfo(name=arcname)
    info.size = len(content)
    tar.addfile(info, io.BytesIO(content))


def _read_member(tar: tarfile.TarFile, name: str) -> bytes:
    """Read one tar member fully, capped at the per-member byte limit."""
    try:
        member = tar.getmember(name)
    except KeyError as exc:
        raise ValueError(f"Bundle member not found: {name}") from exc
    cap = _max_bundle_member_bytes()
    if member.size > cap:
        raise ValueError(
            f"Bundle member {name!r} declares {member.size} bytes, exceeds {cap}-byte cap"
        )
    extracted = tar.extractfile(member)
    if extracted is None:
        raise ValueError(f"Bundle member is not a regular file: {name}")
    return extracted.read()


def pack_notebook_output_bundle(
    bundle_path: Path,
    result_manifest: dict[str, Any],
    output_dir: Path,
) -> None:
    """Pack one harness result directory into a transport bundle."""
    bundle_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "success": bool(result_manifest.get("success", False)),
        "variables": {},
        "stdout_file": "stdout.txt",
        "stderr_file": "stderr.txt",
        "mutation_warnings": result_manifest.get("mutation_warnings", []),
        "error": result_manifest.get("error"),
        "traceback": result_manifest.get("traceback"),
    }

    variables = result_manifest.get("variables", {})
    if not isinstance(variables, dict):
        raise ValueError("Bundle manifest variables must be a dict")

    for var_name, meta in variables.items():
        if not isinstance(meta, dict):
            raise ValueError(f"Variable metadata for {var_name} must be a dict")

        if "error" in meta:
            bundle_manifest["variables"][var_name] = dict(meta)
            continue

        file_name = meta.get("file")
        if not isinstance(file_name, str) or not file_name:
            raise ValueError(f"Variable {var_name} is missing an output file")

        src = output_dir / file_name
        if not src.exists():
            raise ValueError(f"Output file for {var_name} does not exist: {src}")

        bundle_file = f"files/{src.name}"
        bundle_meta = dict(meta)
        bundle_meta["file"] = bundle_file
        bundle_manifest["variables"][var_name] = bundle_meta

    with tarfile.open(bundle_path, "w") as tar:
        _add_bytes(
            tar,
            "manifest.json",
            json.dumps(bundle_manifest, indent=2, sort_keys=True).encode("utf-8"),
        )
        _add_bytes(
            tar,
            "stdout.txt",
            str(result_manifest.get("stdout", "")).encode("utf-8"),
        )
        _add_bytes(
            tar,
            "stderr.txt",
            str(result_manifest.get("stderr", "")).encode("utf-8"),
        )

        for meta in bundle_manifest["variables"].values():
            if not isinstance(meta, dict) or "error" in meta:
                continue
            arcname = meta["file"]
            src = output_dir / Path(arcname).name
            tar.add(src, arcname=arcname)


def unpack_notebook_output_bundle(
    bundle_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Unpack a transport bundle into a harness-style output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(bundle_path, "r") as tar:
        manifest_data = json.loads(_read_member(tar, "manifest.json").decode("utf-8"))

        if manifest_data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported notebook bundle schema: {manifest_data.get('schema_version')!r}"
            )

        stdout_text = ""
        stderr_text = ""
        stdout_file = manifest_data.get("stdout_file")
        stderr_file = manifest_data.get("stderr_file")
        if isinstance(stdout_file, str) and stdout_file:
            stdout_text = _read_member(tar, stdout_file).decode("utf-8")
        if isinstance(stderr_file, str) and stderr_file:
            stderr_text = _read_member(tar, stderr_file).decode("utf-8")

        result: dict[str, Any] = {
            "success": bool(manifest_data.get("success", False)),
            "variables": {},
            "stdout": stdout_text,
            "stderr": stderr_text,
            "mutation_warnings": manifest_data.get("mutation_warnings", []),
            "error": manifest_data.get("error"),
            "traceback": manifest_data.get("traceback"),
        }

        variables = manifest_data.get("variables", {})
        if not isinstance(variables, dict):
            raise ValueError("Bundle manifest variables must be a dict")

        for var_name, meta in variables.items():
            if not isinstance(meta, dict):
                raise ValueError(f"Variable metadata for {var_name} must be a dict")

            if "error" in meta:
                result["variables"][var_name] = dict(meta)
                continue

            bundle_file = meta.get("file")
            if not isinstance(bundle_file, str) or not bundle_file.startswith("files/"):
                raise ValueError(f"Invalid bundle file path for {var_name}: {bundle_file}")

            file_name = Path(bundle_file).name
            dest = output_dir / file_name
            with open(dest, "wb") as dst:
                dst.write(_read_member(tar, bundle_file))

            var_meta = dict(meta)
            var_meta["file"] = file_name
            result["variables"][var_name] = var_meta

    # Match the local harness contract: write to result.json (not
    # manifest.json) so downstream code that reads the executor
    # output from disk finds it under the same name regardless of
    # whether the cell ran via local harness or remote bundle.
    with open(output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def read_notebook_output_bundle_manifest(data: bytes) -> dict[str, Any]:
    """Read and validate the top-level manifest from bundle bytes."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
        manifest_data = json.loads(_read_member(tar, "manifest.json").decode("utf-8"))
        _validate_bundle_manifest(manifest_data, tar)

    return manifest_data


def read_notebook_output_bundle_manifest_path(path: Path) -> dict[str, Any]:
    """Read and validate the top-level manifest from a bundle file on disk."""
    with tarfile.open(path, mode="r") as tar:
        manifest_data = json.loads(_read_member(tar, "manifest.json").decode("utf-8"))
        _validate_bundle_manifest(manifest_data, tar)

    return manifest_data


def _validate_bundle_manifest(
    manifest_data: dict[str, Any],
    tar: tarfile.TarFile,
) -> None:
    """Validate bundle manifest shape and referenced members."""
    if manifest_data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported notebook bundle schema: {manifest_data.get('schema_version')!r}"
        )

    variables = manifest_data.get("variables", {})
    if not isinstance(variables, dict):
        raise ValueError("Bundle manifest variables must be a dict")

    for stream_field in ("stdout_file", "stderr_file"):
        stream_name = manifest_data.get(stream_field)
        if not isinstance(stream_name, str) or not stream_name:
            raise ValueError(f"Bundle manifest is missing {stream_field}")
        _read_member(tar, stream_name)

    for var_name, meta in variables.items():
        if not isinstance(meta, dict):
            raise ValueError(f"Variable metadata for {var_name} must be a dict")

        if "error" in meta:
            continue

        bundle_file = meta.get("file")
        if not isinstance(bundle_file, str) or not bundle_file:
            raise ValueError(f"Variable {var_name} is missing bundle file metadata")

        bundle_path = Path(bundle_file)
        if (
            not bundle_file.startswith("files/")
            or ".." in bundle_path.parts
            or len(bundle_path.parts) < 2
        ):
            raise ValueError(f"Invalid bundle file path for {var_name}: {bundle_file}")

        _read_member(tar, bundle_file)
