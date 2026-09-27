#!/usr/bin/env python3
"""Smoke test for the local Strata service-mode demo stack.

This script expects the local stack from ``docker-compose.service.yml`` to be
running already. It verifies the main service-mode notebook path:

1. admin identity can access the server-managed notebook worker registry
2. normal notebook user cannot access admin worker APIs
3. normal notebook user creates a notebook and assigns an allowed worker
4. notebook execution succeeds over WebSocket through that worker
5. admin disables the worker
6. forced rerun is rejected by service-mode worker policy

Usage:
    docker compose -f docker-compose.service.yml up -d
    uv run python scripts/service_mode_smoke.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlparse, urlunparse

import httpx
from websockets.sync.client import connect as ws_connect


@dataclass(frozen=True)
class SmokeConfig:
    user_base_url: str
    admin_base_url: str
    worker_name: str
    timeout: float


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _require_status(response: httpx.Response, expected: int, context: str) -> None:
    if response.status_code != expected:
        raise RuntimeError(
            f"{context} failed: expected HTTP {expected}, "
            f"got {response.status_code}: {response.text}"
        )


def _wait_for_health(base_url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    health_url = f"{base_url.rstrip('/')}/health"
    last_error = "unreachable"

    while time.time() < deadline:
        try:
            response = httpx.get(health_url, timeout=2.0)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:  # pragma: no cover - transport detail only
            last_error = str(exc)
        time.sleep(0.5)

    raise RuntimeError(f"Timed out waiting for {health_url}: {last_error}")


def _base_to_ws(base_url: str, path: str) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse((scheme, parsed.netloc, path, "", "", ""))


def _ws_send(ws, seq: int, msg_type: str, payload: dict) -> None:
    ws.send(
        json.dumps(
            {
                "type": msg_type,
                "seq": seq,
                "ts": _utc_now_iso(),
                "payload": payload,
            }
        )
    )


def _run_cell(ws, run_type: str, *, cell_id: str, timeout: float) -> dict:
    """Send *run_type* for the cell and return its terminal frame.

    A new notebook syncs its environment first, and a run sent during the
    sync is refused with ENVIRONMENT_BUSY; send it again once the sync ends.
    """
    seq = 1
    _ws_send(ws, seq, run_type, {"cell_id": cell_id})
    deadline = time.time() + timeout
    waiting_for_environment = False

    while time.time() < deadline:
        remaining = max(0.1, deadline - time.time())
        message = json.loads(ws.recv(timeout=remaining))
        msg_type = message.get("type")
        payload = message.get("payload", {})

        if msg_type == "error" and payload.get("code") == "ENVIRONMENT_BUSY":
            waiting_for_environment = True
            continue

        if msg_type == "environment_job_finished" and waiting_for_environment:
            waiting_for_environment = False
            seq += 1
            _ws_send(ws, seq, run_type, {"cell_id": cell_id})
            continue

        if msg_type == "cascade_prompt":
            seq += 1
            _ws_send(
                ws,
                seq,
                "cell_execute_cascade",
                {"cell_id": cell_id, "plan_id": payload["plan_id"]},
            )
            continue

        if msg_type in {"cell_output", "cell_error"} and payload.get("cell_id") == cell_id:
            return message

    raise RuntimeError(f"Timed out waiting for execution result for cell {cell_id}")


def run_smoke(config: SmokeConfig) -> None:
    print("Waiting for proxy endpoints...")
    _wait_for_health(config.user_base_url, config.timeout)
    _wait_for_health(config.admin_base_url, config.timeout)

    user = httpx.Client(base_url=config.user_base_url, timeout=config.timeout)
    admin = httpx.Client(base_url=config.admin_base_url, timeout=config.timeout)

    worker_disabled = False

    try:
        print("Checking admin worker registry access...")
        admin_workers = admin.get("/v1/admin/notebook-workers")
        _require_status(admin_workers, 200, "admin worker registry lookup")
        worker_names = {worker["name"] for worker in admin_workers.json()["workers"]}
        if config.worker_name not in worker_names:
            raise RuntimeError(
                f"Expected worker '{config.worker_name}' in admin catalog, "
                f"got {sorted(worker_names)}"
            )

        print("Checking non-admin worker registry rejection...")
        user_admin_workers = user.get("/v1/admin/notebook-workers")
        _require_status(user_admin_workers, 403, "user access to admin worker registry")

        print("Checking service-mode session discovery is blocked...")
        sessions = user.get("/v1/notebooks/sessions")
        _require_status(sessions, 403, "service-mode session listing")

        notebook_name = f"service-smoke-{uuid.uuid4().hex[:8]}"
        print(f"Creating notebook '{notebook_name}'...")
        created = user.post(
            "/v1/notebooks/create",
            json={"parent_path": "/tmp/strata-notebooks", "name": notebook_name},
        )
        _require_status(created, 200, "create notebook")
        notebook = created.json()
        notebook_id = notebook["session_id"]

        print("Adding a cell...")
        cell_response = user.post(
            f"/v1/notebooks/{notebook_id}/cells",
            json={"after_cell_id": None},
        )
        _require_status(cell_response, 200, "add notebook cell")
        cell_id = cell_response.json()["id"]

        print(f"Assigning worker '{config.worker_name}'...")
        assigned = user.put(
            f"/v1/notebooks/{notebook_id}/worker",
            json={"worker": config.worker_name},
        )
        _require_status(assigned, 200, "assign notebook worker")

        print("Updating cell source...")
        updated = user.put(
            f"/v1/notebooks/{notebook_id}/cells/{cell_id}",
            json={"source": "x = 1"},
        )
        _require_status(updated, 200, "update cell source")

        ws_url = _base_to_ws(config.user_base_url, f"/v1/notebooks/ws/{notebook_id}")
        print("Executing allowed worker over WebSocket...")
        with ws_connect(ws_url, open_timeout=config.timeout) as ws:
            result = _run_cell(ws, "cell_execute", cell_id=cell_id, timeout=config.timeout)

        if result["type"] != "cell_output":
            raise RuntimeError(f"Expected successful cell_output, got {result}")
        payload = result["payload"]
        if payload.get("remote_worker") != config.worker_name:
            raise RuntimeError(f"Expected remote_worker={config.worker_name}, got {payload}")
        if payload.get("outputs", {}).get("x", {}).get("preview") != 1:
            raise RuntimeError(f"Unexpected cell output payload: {payload}")

        print(f"Disabling worker '{config.worker_name}' as admin...")
        disabled = admin.patch(
            f"/v1/admin/notebook-workers/{config.worker_name}",
            json={"enabled": False},
        )
        _require_status(disabled, 200, "disable admin notebook worker")
        worker_disabled = True

        print("Verifying forced rerun is rejected by service-mode policy...")
        with ws_connect(ws_url, open_timeout=config.timeout) as ws:
            blocked = _run_cell(ws, "cell_execute_force", cell_id=cell_id, timeout=config.timeout)

        if blocked["type"] != "cell_error":
            raise RuntimeError(f"Expected blocked cell_error, got {blocked}")
        error_text = blocked["payload"].get("error", "")
        if "disabled by server policy" not in error_text:
            raise RuntimeError(f"Unexpected blocked execution error: {error_text}")

        print("Smoke test passed.")
    finally:
        if worker_disabled:
            try:
                admin.patch(
                    f"/v1/admin/notebook-workers/{config.worker_name}",
                    json={"enabled": True},
                )
            except Exception:
                pass
        user.close()
        admin.close()


def parse_args() -> SmokeConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-base-url", default="http://127.0.0.1:8865")
    parser.add_argument("--admin-base-url", default="http://127.0.0.1:8866")
    parser.add_argument("--worker", default="gpu-http")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    return SmokeConfig(
        user_base_url=args.user_base_url.rstrip("/"),
        admin_base_url=args.admin_base_url.rstrip("/"),
        worker_name=args.worker,
        timeout=args.timeout,
    )


def main() -> int:
    try:
        run_smoke(parse_args())
    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
