"""HTTP executor app for notebook cell execution."""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import os
import shutil
import socket
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.background import BackgroundTask

from strata.blob_store import BLOB_STREAM_CHUNK_BYTES
from strata.notebook.models import MountSpec
from strata.notebook.mounts import MountResolver, parse_mount_uri
from strata.notebook.remote_bundle import pack_notebook_output_bundle
from strata.types import EXECUTOR_PROTOCOL_HEADER, EXECUTOR_PROTOCOL_VERSION

NOTEBOOK_EXECUTOR_PROTOCOL_VERSION = "notebook-cell-v1"
NOTEBOOK_EXECUTOR_TRANSFORM_REF = "notebook_cell@v1"
NOTEBOOK_EXECUTOR_MANIFEST_VERSION = "notebook-build-manifest@v1"

# --- Signed-URL manifest trust-model defenses ---------------------------------
#
# The /v1/execute-manifest endpoint accepts a manifest of pre-signed URLs
# the worker must fetch (inputs) or POST to (upload, finalize). The whole
# v2 pull model's security story is "those URLs are signed and short-lived",
# but the worker can't verify the signature itself — it only sees the URL.
# A compromised or buggy orchestrator could hand the worker URLs that
# point at internal services (SSRF) or unbounded streams (OOM). The
# defenses below are cheap and don't depend on the orchestrator behaving
# correctly.

_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})

# Per-input download cap. Override via STRATA_WORKER_MAX_INPUT_BYTES.
_DEFAULT_MAX_INPUT_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB


def _max_input_bytes() -> int:
    raw = os.environ.get("STRATA_WORKER_MAX_INPUT_BYTES")
    if not raw:
        return _DEFAULT_MAX_INPUT_BYTES
    try:
        parsed = int(raw)
    except ValueError:
        return _DEFAULT_MAX_INPUT_BYTES
    return parsed if parsed > 0 else _DEFAULT_MAX_INPUT_BYTES


def _allow_local_hosts() -> bool:
    """Whether to bypass the host-IP SSRF check.

    Default off. Tests and local development that need to fetch from
    127.0.0.1 / 192.168.x.x / docker-bridge addresses set
    ``STRATA_WORKER_ALLOW_LOCAL_HOSTS=1``. Production worker
    deployments should leave it unset so the SSRF defense is active.
    """
    return os.environ.get("STRATA_WORKER_ALLOW_LOCAL_HOSTS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _assert_url_safe(url: str, field: str) -> None:
    """Reject manifest URLs that are scheme- or host-unsafe.

    A compromised or buggy orchestrator could hand the worker URLs
    that point at internal services. Two distinct defenses:

    1. **Scheme allowlist** — only http and https. Blocks file://,
       data:, javascript:, ftp:// and any other scheme httpx might
       grow plugin support for.
    2. **Host resolution + IP-range blocklist** — the resolved IP
       must not be loopback, link-local (incl. cloud metadata
       169.254.169.254 / fd00:ec2::254), private, multicast,
       reserved, or unspecified. Hostnames are resolved via
       getaddrinfo and every returned address is checked; a
       hostname that resolves to multiple addresses must have all
       of them in the public range to pass. This rules out both
       direct internal-IP URLs and hostname-based variants
       (e.g. metadata.google.internal). Set
       ``STRATA_WORKER_ALLOW_LOCAL_HOSTS=1`` to bypass the IP check
       (tests / local dev with 127.0.0.1 build servers); production
       deployments leave it unset.

    Allowlist-on-host instead of blocklist-on-host would be more
    restrictive but breaks real signed-URL usage where S3/GCS
    buckets resolve to public IPs across many regions. Blocklist
    on internal ranges is the right tradeoff.

    Caveats:
    * DNS rebinding race: the IP we resolved here may differ from
      the IP httpx resolves at fetch time. Honest mitigation
      requires resolving once and passing the IP to httpx; left as
      a follow-up because the practical attacker who controls DNS
      already has stronger primitives.
    * IPv4-mapped IPv6 (``::ffff:127.0.0.1``) is caught — we
      ``unmap()`` before checking.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_URL_SCHEMES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Manifest {field} URL uses disallowed scheme {scheme!r}; "
                f"only {sorted(_ALLOWED_URL_SCHEMES)} are accepted."
            ),
        )

    if _allow_local_hosts():
        return

    host = parsed.hostname
    if not host:
        raise HTTPException(
            status_code=400,
            detail=f"Manifest {field} URL is missing a host: {url!r}",
        )

    try:
        addrinfo = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Manifest {field} URL host {host!r} did not resolve: {exc}",
        ) from exc

    for entry in addrinfo:
        sockaddr = entry[4]
        try:
            ip = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Manifest {field} URL host {host!r} resolved to non-IP address {sockaddr[0]!r}"
                ),
            ) from None
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        if (
            ip.is_loopback
            or ip.is_link_local
            or ip.is_private
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Manifest {field} URL host {host!r} resolves to "
                    f"non-routable address {ip}; refusing to fetch"
                ),
            )


async def _run_harness(
    harness_path: Path,
    manifest_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run the notebook harness with one manifest file."""
    from strata.notebook.process_tree import (
        subprocess_kwargs_for_new_group,
        terminate_subprocess_tree,
    )

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(harness_path),
        str(manifest_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **subprocess_kwargs_for_new_group(),
    )
    try:
        _stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        await terminate_subprocess_tree(proc)
        raise TimeoutError()

    result_path = manifest_path.parent / "harness-result.json"
    if not result_path.exists():
        raise RuntimeError(f"Harness did not produce harness-result.json: {stderr.decode()}")

    with open(result_path, encoding="utf-8") as f:
        return json.load(f)


def create_notebook_executor_app() -> FastAPI:
    """Create a standalone notebook executor HTTP app.

    Optional bearer-token auth via ``STRATA_WORKER_TOKEN`` env var. When
    set, the ``/v1/*`` execution endpoints require
    ``Authorization: Bearer <token>``. ``/health`` stays open so platform
    health probes (Fly, Cloudflare, k8s liveness) don't need the secret.
    Unset = no auth, backward-compatible with existing deployments.
    """
    started_at = time.time()
    active_executions = 0

    # ---- Bearer-token gate ----
    expected_token = os.environ.get("STRATA_WORKER_TOKEN", "").strip() or None

    async def require_worker_token(http_request: Request) -> None:
        if expected_token is None:
            return  # Auth disabled
        header = http_request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Missing or malformed Authorization header (expected Bearer token)",
            )
        presented = header[len("Bearer ") :]
        # Constant-time compare; the public-facing comparison shouldn't
        # leak token length via timing.
        if not hmac.compare_digest(presented, expected_token):
            raise HTTPException(status_code=401, detail="Invalid worker token")

    def _input_extension(content_type: str) -> str:
        return {
            "arrow/ipc": ".arrow",
            "json/object": ".json",
            "pickle/object": ".pickle",
            "module/import": ".module.json",
            "module/cell": ".cell_module.json",
            "module/cell-instance": ".cell_instance.pickle",
        }.get(content_type, ".bin")

    def _response_error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except Exception:
            payload = None

        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("error")
            if detail:
                return str(detail)

        text = response.text.strip()
        return text or f"HTTP {response.status_code}"

    async def _execute_to_bundle(
        *,
        source: str,
        timeout_seconds: float,
        raw_inputs: dict[str, dict[str, Any]],
        raw_mounts: list[dict[str, Any]],
        runtime_env: dict[str, str],
        write_input_bytes: Any,
    ) -> tuple[Path, Path] | JSONResponse:
        """Execute a cell and pack outputs into a bundle file.

        On success, returns ``(bundle_path, tmpdir)`` — the caller is
        responsible for deleting ``tmpdir`` after the bundle bytes have
        been consumed. On failure, returns a ``JSONResponse`` with the
        tmpdir already cleaned up.
        """
        if not isinstance(raw_inputs, dict):
            raise HTTPException(status_code=400, detail="inputs must be an object")
        if not isinstance(raw_mounts, list):
            raise HTTPException(status_code=400, detail="mounts must be a list")

        mount_specs = [MountSpec(**mount) for mount in raw_mounts]
        for mount in mount_specs:
            scheme, _path = parse_mount_uri(mount.uri)
            if scheme == "file":
                raise HTTPException(
                    status_code=400,
                    detail=(f"Remote execution does not support file:// mount '{mount.name}'"),
                )

        nonlocal active_executions

        tmpdir = Path(tempfile.mkdtemp(prefix="strata_notebook_executor_"))
        try:
            output_dir = tmpdir

            inputs: dict[str, dict[str, str]] = {}
            for var_name, spec in raw_inputs.items():
                if not isinstance(spec, dict):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Input spec for {var_name} must be an object",
                    )
                content_type = str(spec.get("content_type", "pickle/object"))
                requested_file_name = Path(str(spec.get("file", ""))).name
                file_name = requested_file_name or f"{var_name}{_input_extension(content_type)}"
                data = await write_input_bytes(var_name, file_name, spec)
                with open(output_dir / file_name, "wb") as f:
                    f.write(data)
                inputs[var_name] = {
                    "content_type": content_type,
                    "file": file_name,
                }

            mount_resolver = MountResolver(
                cache_dir=output_dir / "mount_cache",
            )
            resolved_mounts = await mount_resolver.prepare_mounts(mount_specs)
            manifest_mounts = {
                name: {
                    "uri": rm.spec.uri,
                    "mode": rm.spec.mode.value,
                    "local_path": str(rm.local_path),
                }
                for name, rm in resolved_mounts.items()
            }

            manifest = {
                "source": source,
                "inputs": inputs,
                "output_dir": str(output_dir),
                "mounts": manifest_mounts,
                "env": runtime_env,
            }
            manifest_path = output_dir / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)

            harness_path = Path(__file__).parent / "harness.py"
            active_executions += 1
            try:
                result = await _run_harness(
                    harness_path,
                    manifest_path,
                    timeout_seconds,
                )
                if result.get("success", False):
                    await mount_resolver.sync_back(resolved_mounts)
            except TimeoutError:
                shutil.rmtree(tmpdir, ignore_errors=True)
                # Lazy import: keep the heavy executor module out of the worker's
                # module-load; this is a rare error path.
                from strata.notebook.executor import cell_timeout_message

                return JSONResponse(
                    status_code=408,
                    content={
                        "success": False,
                        "error": cell_timeout_message(timeout_seconds),
                    },
                )
            except Exception as exc:
                shutil.rmtree(tmpdir, ignore_errors=True)
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "error": str(exc)},
                )
            finally:
                active_executions -= 1

            bundle_path = output_dir / "notebook-output-bundle.tar"
            pack_notebook_output_bundle(bundle_path, result, output_dir)
            return bundle_path, tmpdir
        except BaseException:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise

    async def _run_notebook_execution(
        *,
        source: str,
        timeout_seconds: float,
        raw_inputs: dict[str, dict[str, Any]],
        raw_mounts: list[dict[str, Any]],
        runtime_env: dict[str, str],
        form: Any,
    ) -> Response:
        async def _write_uploaded_input(
            var_name: str,
            requested_file_name: str,
            _spec: dict[str, Any],
        ) -> bytes:
            upload = form.get(var_name) or form.get(requested_file_name)
            if upload is None or isinstance(upload, str):
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing uploaded input file: {var_name}",
                )
            return await upload.read()

        result = await _execute_to_bundle(
            source=source,
            timeout_seconds=timeout_seconds,
            raw_inputs=raw_inputs,
            raw_mounts=raw_mounts,
            runtime_env=runtime_env,
            write_input_bytes=_write_uploaded_input,
        )
        if isinstance(result, JSONResponse):
            return result
        bundle_path, tmpdir = result
        return FileResponse(
            path=bundle_path,
            media_type="application/x-tar",
            headers={
                "X-Strata-Notebook-Executor-Protocol": NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
                EXECUTOR_PROTOCOL_HEADER: EXECUTOR_PROTOCOL_VERSION,
            },
            background=BackgroundTask(shutil.rmtree, tmpdir, True),
        )

    app = FastAPI(
        title="Strata Notebook Executor",
        description="Reference notebook executor for remote notebook workers",
        version="1.0.0",
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "healthy",
            "capabilities": {
                "protocol_versions": [EXECUTOR_PROTOCOL_VERSION],
                "transform_refs": [NOTEBOOK_EXECUTOR_TRANSFORM_REF],
                "features": {
                    "notebook_protocol_version": NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
                    "output_format": "notebook-output-bundle@v1",
                    "pull_model": True,
                },
            },
            "version": "1.0.0",
            "uptime_seconds": max(0.0, time.time() - started_at),
            "active_executions": active_executions,
        }

    @app.post("/v1/notebook-execute", dependencies=[Depends(require_worker_token)])
    async def execute(http_request: Request) -> Response:
        form = await http_request.form()
        metadata_file = form.get("metadata")
        if metadata_file is None or isinstance(metadata_file, str):
            raise HTTPException(status_code=400, detail="Missing metadata")

        try:
            metadata = json.loads((await metadata_file.read()).decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid metadata: {exc}")

        protocol_version = metadata.get("protocol_version")
        if protocol_version != NOTEBOOK_EXECUTOR_PROTOCOL_VERSION:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported protocol version: {protocol_version}. "
                    f"Expected: {NOTEBOOK_EXECUTOR_PROTOCOL_VERSION}"
                ),
            )

        source = str(metadata.get("source", ""))
        timeout_seconds = float(metadata.get("timeout_seconds", 30.0))
        raw_inputs = metadata.get("inputs", {})
        raw_mounts = metadata.get("mounts", [])
        runtime_env = metadata.get("env", {})

        return await _run_notebook_execution(
            source=source,
            timeout_seconds=timeout_seconds,
            raw_inputs=raw_inputs,
            raw_mounts=raw_mounts,
            runtime_env=runtime_env,
            form=form,
        )

    @app.post("/v1/execute", dependencies=[Depends(require_worker_token)])
    async def execute_protocol_v1(http_request: Request) -> Response:
        """Execute notebook cells using the standard executor v1 metadata envelope."""
        form = await http_request.form()
        metadata_file = form.get("metadata")
        if metadata_file is None or isinstance(metadata_file, str):
            raise HTTPException(status_code=400, detail="Missing metadata")

        try:
            metadata = json.loads((await metadata_file.read()).decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid metadata: {exc}")

        protocol_version = metadata.get("protocol_version", EXECUTOR_PROTOCOL_VERSION)
        if protocol_version != EXECUTOR_PROTOCOL_VERSION:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported protocol version: {protocol_version}. "
                    f"Expected: {EXECUTOR_PROTOCOL_VERSION}"
                ),
            )

        transform = metadata.get("transform", {})
        transform_ref = str(transform.get("ref", ""))
        if transform_ref != NOTEBOOK_EXECUTOR_TRANSFORM_REF:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported transform: {transform_ref}. "
                    f"Expected: {NOTEBOOK_EXECUTOR_TRANSFORM_REF}"
                ),
            )

        params = transform.get("params", {})
        source = str(params.get("source", ""))
        timeout_seconds = float(params.get("timeout_seconds", 30.0))
        raw_mounts = params.get("mounts", [])
        runtime_env = params.get("env", {})
        input_descriptors = metadata.get("inputs", [])

        if not isinstance(input_descriptors, list):
            raise HTTPException(status_code=400, detail="inputs must be a list")

        raw_inputs: dict[str, dict[str, Any]] = {}
        for descriptor in input_descriptors:
            if not isinstance(descriptor, dict):
                raise HTTPException(status_code=400, detail="input descriptor must be an object")
            name = str(descriptor.get("name", "")).strip()
            if not name:
                raise HTTPException(status_code=400, detail="input descriptor missing name")
            content_type = str(descriptor.get("format", "pickle/object"))
            raw_inputs[name] = {
                "content_type": content_type,
                "file": f"{name}{_input_extension(content_type)}",
            }

        return await _run_notebook_execution(
            source=source,
            timeout_seconds=timeout_seconds,
            raw_inputs=raw_inputs,
            raw_mounts=raw_mounts,
            runtime_env=runtime_env,
            form=form,
        )

    @app.post("/v1/execute-manifest", dependencies=[Depends(require_worker_token)])
    async def execute_manifest(http_request: Request) -> Response:
        """Execute a notebook build from a signed manifest."""
        try:
            manifest = await http_request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid manifest payload: {exc}")

        if not isinstance(manifest, dict):
            raise HTTPException(status_code=400, detail="Manifest payload must be an object")

        metadata = manifest.get("metadata", {})
        if not isinstance(metadata, dict):
            raise HTTPException(status_code=400, detail="Manifest metadata must be an object")

        executor_ref = str(metadata.get("executor_ref", ""))
        if executor_ref != NOTEBOOK_EXECUTOR_TRANSFORM_REF:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported executor ref: {executor_ref}. "
                    f"Expected: {NOTEBOOK_EXECUTOR_TRANSFORM_REF}"
                ),
            )

        params = metadata.get("params", {})
        if not isinstance(params, dict):
            raise HTTPException(status_code=400, detail="Manifest params must be an object")

        raw_inputs = params.get("input_specs", {})
        raw_mounts = params.get("mounts", [])
        runtime_env = params.get("env", {})
        source = str(params.get("source", ""))
        timeout_seconds = float(params.get("timeout_seconds", 30.0))

        input_urls = manifest.get("inputs", [])
        if not isinstance(input_urls, list):
            raise HTTPException(status_code=400, detail="Manifest inputs must be a list")

        input_url_by_uri: dict[str, str] = {}
        for item in input_urls:
            if not isinstance(item, dict):
                raise HTTPException(
                    status_code=400,
                    detail="Manifest input entry must be an object",
                )
            artifact_id = str(item.get("artifact_id", "")).strip()
            version = item.get("version")
            url = str(item.get("url", "")).strip()
            if not artifact_id or not url or not isinstance(version, int):
                raise HTTPException(status_code=400, detail="Manifest input entry is incomplete")
            _assert_url_safe(url, f"input[{artifact_id}@v={version}]")
            input_url_by_uri[f"strata://artifact/{artifact_id}@v={version}"] = url

        output = manifest.get("output", {})
        if not isinstance(output, dict):
            raise HTTPException(status_code=400, detail="Manifest output must be an object")
        upload_url = str(output.get("url", "")).strip()
        finalize_url = str(manifest.get("finalize_url", "")).strip()
        if not upload_url or not finalize_url:
            raise HTTPException(status_code=400, detail="Manifest is missing upload/finalize URLs")
        _assert_url_safe(upload_url, "output.url")
        _assert_url_safe(finalize_url, "finalize_url")

        async def _download_input(
            var_name: str,
            _requested_file_name: str,
            spec: dict[str, Any],
        ) -> bytes:
            input_uri = str(spec.get("uri", "")).strip()
            if not input_uri:
                raise HTTPException(
                    status_code=400,
                    detail=f"Manifest input spec for {var_name} is missing uri",
                )
            download_url = input_url_by_uri.get(input_uri)
            if download_url is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Manifest does not include a signed URL for {input_uri}",
                )
            # Stream + cap so a misconfigured-or-malicious download
            # can't OOM the worker. Content-Length (when present) lets
            # us reject up front before reading any bytes.
            max_bytes = _max_input_bytes()
            async with httpx.AsyncClient(timeout=max(timeout_seconds, 30.0)) as client:
                async with client.stream("GET", download_url) as response:
                    if response.status_code != 200:
                        raise HTTPException(
                            status_code=502,
                            detail=(
                                f"Failed to download notebook input {input_uri}: "
                                f"{response.status_code}"
                            ),
                        )
                    declared = response.headers.get("content-length")
                    if declared is not None:
                        try:
                            declared_bytes = int(declared)
                        except ValueError:
                            declared_bytes = -1
                        if declared_bytes > max_bytes:
                            raise HTTPException(
                                status_code=413,
                                detail=(
                                    f"Notebook input {input_uri} declared "
                                    f"{declared_bytes} bytes, exceeds {max_bytes}-byte cap"
                                ),
                            )
                    buf = bytearray()
                    async for chunk in response.aiter_bytes():
                        buf.extend(chunk)
                        if len(buf) > max_bytes:
                            raise HTTPException(
                                status_code=413,
                                detail=(
                                    f"Notebook input {input_uri} exceeds "
                                    f"{max_bytes}-byte cap during download"
                                ),
                            )
            return bytes(buf)

        bundle_result = await _execute_to_bundle(
            source=source,
            timeout_seconds=timeout_seconds,
            raw_inputs=raw_inputs,
            raw_mounts=raw_mounts,
            runtime_env=runtime_env,
            write_input_bytes=_download_input,
        )
        if isinstance(bundle_result, JSONResponse):
            return bundle_result

        bundle_path, tmpdir = bundle_result
        try:
            byte_size = bundle_path.stat().st_size

            async def _stream_bundle_body() -> AsyncIterator[bytes]:
                with open(bundle_path, "rb") as f:
                    while chunk := f.read(BLOB_STREAM_CHUNK_BYTES):
                        yield chunk

            try:
                async with httpx.AsyncClient(timeout=max(timeout_seconds, 30.0)) as client:
                    upload_response = await client.post(
                        upload_url,
                        content=_stream_bundle_body(),
                        headers={
                            "Content-Type": "application/x-tar",
                            "Content-Length": str(byte_size),
                        },
                    )
                    if upload_response.status_code != 200:
                        raise HTTPException(
                            status_code=502,
                            detail=(
                                "Failed to upload notebook bundle output: "
                                f"{upload_response.status_code} "
                                f"({_response_error_detail(upload_response)})"
                            ),
                        )

                    finalize_response = await client.post(
                        finalize_url,
                        json={"output_format": "notebook-output-bundle@v1"},
                    )
            except httpx.TimeoutException as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Notebook bundle transfer timed out: {exc}",
                ) from exc
            except httpx.HTTPError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Notebook bundle transfer failed: {exc}",
                ) from exc

            if finalize_response.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Failed to finalize notebook bundle build: "
                        f"{finalize_response.status_code} "
                        f"({_response_error_detail(finalize_response)})"
                    ),
                )

            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "build_id": manifest.get("build_id"),
                    "byte_size": byte_size,
                    "protocol_version": NOTEBOOK_EXECUTOR_MANIFEST_VERSION,
                    "finalize": finalize_response.json(),
                },
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return app


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point: run the notebook executor HTTP app.

    Used by ``python -m strata.notebook.remote_executor --port 9000`` and
    by deployment images that run a single executor process. For local
    multi-worker testing, launch multiple instances on different ports.
    """
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        prog="strata-worker",
        description=(
            "Run a Strata notebook worker — an HTTP endpoint that accepts "
            "cells and returns their outputs. Cells run in the Python "
            "environment this process was started in, so install your "
            "workload dependencies (pandas, torch, datafusion, ...) before "
            "launching."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9000, help="Bind port (default: 9000)")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Uvicorn log level",
    )
    args = parser.parse_args(argv)

    app = create_notebook_executor_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
