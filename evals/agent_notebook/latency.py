"""The scratchpad's latency, as an agent meets it: first use, then each trivial cell.

On a fresh project this starts ``strata mcp`` over stdio, as an agent's config does, and
times its first call (which creates the scratchpad and starts its daemon, reported step
by step in ``setup_ms``), then *n* trivial uncached cells through ``run``, then the same
through the CLI (``strata cell add --run`` on the scratchpad, which goes through the
daemon). The phase-2 goals: p95 under 0.3 s through MCP and under 1 s through the CLI,
and a first call within 3 s of Strata overhead.

Strata's stores and the registry go to a temporary home; uv keeps the real cache unless
``--cold-uv-cache``, which gives it an empty one, as on a machine that never synced.

    uv run python -m evals.agent_notebook.latency --calls 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from mcp import Client
from mcp.client.stdio import StdioServerParameters


def _uv_dir(kind: str) -> str:
    return subprocess.run(
        ["uv", kind, "dir"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _env(base: Path, cold_uv_cache: bool) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
    env.update(
        # Read before HOME moves: uv's cache and Pythons live under the real one.
        UV_CACHE_DIR=str(base / "uv-cache") if cold_uv_cache else _uv_dir("cache"),
        UV_PYTHON_INSTALL_DIR=_uv_dir("python"),
        HOME=str(base / "home"),
        STRATA_CACHE_DIR=str(base / "cache"),
        STRATA_ARTIFACT_DIR=str(base / "artifacts"),
        STRATA_IDLE_EXIT_MINUTES="2",
    )
    return env


def _summary(seconds: list[float]) -> dict[str, float]:
    ordered = sorted(seconds)
    return {
        "calls": len(ordered),
        "p50_ms": round(statistics.median(ordered) * 1000),
        "p95_ms": round(ordered[max(0, round(0.95 * len(ordered)) - 1)] * 1000),
        "max_ms": round(ordered[-1] * 1000),
    }


async def _through_mcp(project: Path, env: dict[str, str], calls: int) -> dict[str, Any]:
    server = StdioServerParameters(
        command=sys.executable, args=["-m", "strata.cli", "mcp"], env=env, cwd=str(project)
    )
    started = time.monotonic()
    async with Client(server, mode="legacy") as client:
        await client.list_tools()
        listed = time.monotonic() - started
        began = time.monotonic()
        first = await client.call_tool("run", {"source": "1 + 1"})
        first_s = time.monotonic() - began
        assert first.structured_content["status"] == "ok", first.structured_content
        timings = []
        for i in range(calls):
            began = time.monotonic()
            result = await client.call_tool("run", {"source": f"{i} * 2  # mcp {i}"})
            timings.append(time.monotonic() - began)
            value = result.structured_content
            assert value["status"] == "ok" and not value.get("cache_hit"), value
    setup_ms = first.structured_content.get("setup_ms", {})
    return {
        "tools_listed_ms": round(listed * 1000),
        "first_call_ms": round(first_s * 1000),
        "setup_ms": setup_ms,
        "first_cell_ms": round(first_s * 1000) - sum(setup_ms.values()),
        "uncached": _summary(timings),
    }


def _through_cli(project: Path, env: dict[str, str], calls: int) -> dict[str, Any]:
    notebook = str(project / ".strata-scratch")
    timings = []
    for i in range(calls):
        began = time.monotonic()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "strata.cli",
                "cell",
                "add",
                notebook,
                "-c",
                f"{i} * 3  # cli {i}",
                "--run",
            ],
            env=env,
            capture_output=True,
            check=True,
        )
        timings.append(time.monotonic() - began)
    return {"uncached": _summary(timings)}


def measure(calls: int, cold_uv_cache: bool) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="strata-latency-") as tmp:
        base = Path(tmp)
        project = base / "project"
        project.mkdir()
        env = _env(base, cold_uv_cache)
        try:
            mcp = asyncio.run(_through_mcp(project, env, calls))
            cli = _through_cli(project, env, calls)
        finally:
            subprocess.run(
                [sys.executable, "-m", "strata.cli", "scratch", "stop", "--root", str(project)],
                env=env,
                capture_output=True,
            )
    return {"cold_uv_cache": cold_uv_cache, "mcp": mcp, "cli": cli}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--cold-uv-cache", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(measure(args.calls, args.cold_uv_cache), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
