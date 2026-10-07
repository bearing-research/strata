"""Bundle helpers for notebook remote-style execution results."""

from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any, BinaryIO

from strata.notebook.harness_user import open_run_file, write_run_file

SCHEMA_VERSION = "notebook-output-bundle@v1"

# Per-member cap: a member is read fully into memory on unpack, so an
# unbounded one can OOM. Override via STRATA_NOTEBOOK_MAX_BUNDLE_MEMBER_BYTES.
_DEFAULT_MAX_BUNDLE_MEMBER_BYTES = 2 * 1024 * 1024 * 1024


def _max_bundle_member_bytes() -> int:
    raw = os.environ.get("STRATA_NOTEBOOK_MAX_BUNDLE_MEMBER_BYTES")
    if not raw:
        return _DEFAULT_MAX_BUNDLE_MEMBER_BYTES
    try:
        parsed = int(raw)
    except ValueError:
        return _DEFAULT_MAX_BUNDLE_MEMBER_BYTES
    return parsed if parsed > 0 else _DEFAULT_MAX_BUNDLE_MEMBER_BYTES


def _open_tar(bundle: Path | BinaryIO, mode: str) -> tarfile.TarFile:
    if isinstance(bundle, Path):
        return tarfile.open(bundle, mode)
    return tarfile.open(fileobj=bundle, mode=mode)


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
    bundle_path: Path | BinaryIO,
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
        # The worker's interpreter and hardware, not ours: a remote cell is
        # exactly where the two differ.
        "build_env": result_manifest.get("build_env", ""),
    }
    # Only a worker reports it; a local bundle omits it rather than claim an empty machine.
    if result_manifest.get("hardware"):
        bundle_manifest["hardware"] = result_manifest["hardware"]

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

    # Every display, not only the last (which also travels as ``_``).
    raw_displays = result_manifest.get("displays", [])
    if not isinstance(raw_displays, list):
        raise ValueError("Bundle manifest displays must be a list")
    bundle_manifest["displays"] = []
    for display in raw_displays:
        if not isinstance(display, dict):
            raise ValueError("Each display must be a dict")
        bundle_display = dict(display)
        file_name = display.get("file")
        if "error" not in display and isinstance(file_name, str) and file_name:
            src = output_dir / file_name
            if not src.exists():
                raise ValueError(f"Output file for a display does not exist: {src}")
            bundle_display["file"] = f"files/{src.name}"
        bundle_manifest["displays"].append(bundle_display)

    with _open_tar(bundle_path, "w") as tar:
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

        # The last display is also ``_``, so the same file is named twice.
        added: set[str] = set()
        for meta in [*bundle_manifest["variables"].values(), *bundle_manifest["displays"]]:
            if not isinstance(meta, dict) or "error" in meta:
                continue
            arcname = meta.get("file")
            if not isinstance(arcname, str) or arcname in added:
                continue
            added.add(arcname)
            with open_run_file(output_dir, Path(arcname).name) as f:
                tar.addfile(tar.gettarinfo(arcname=arcname, fileobj=f), f)


def unpack_notebook_output_bundle(
    bundle_path: Path | BinaryIO,
    output_dir: Path,
) -> dict[str, Any]:
    """Unpack a transport bundle into a harness-style output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with _open_tar(bundle_path, "r") as tar:
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
            "build_env": manifest_data.get("build_env", ""),
            "hardware": manifest_data.get("hardware") or {},
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
            write_run_file(output_dir, file_name, _read_member(tar, bundle_file))

            var_meta = dict(meta)
            var_meta["file"] = file_name
            result["variables"][var_name] = var_meta

        displays = manifest_data.get("displays", [])
        if not isinstance(displays, list):
            raise ValueError("Bundle manifest displays must be a list")
        result["displays"] = []
        for display in displays:
            if not isinstance(display, dict):
                raise ValueError("Each display must be a dict")
            unpacked = dict(display)
            bundle_file = display.get("file")
            if "error" not in display and bundle_file is not None:
                if not isinstance(bundle_file, str) or not bundle_file.startswith("files/"):
                    raise ValueError(f"Invalid bundle file path for a display: {bundle_file}")
                file_name = Path(bundle_file).name
                if not (output_dir / file_name).exists():
                    write_run_file(output_dir, file_name, _read_member(tar, bundle_file))
                unpacked["file"] = file_name
            result["displays"].append(unpacked)

    # Same name as the local harness, so readers find the output either way.
    # Hyphenated so it can't collide with a user variable named ``result``.
    write_run_file(
        output_dir, "harness-result.json", json.dumps(result, indent=2).encode("utf-8")
    )

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
