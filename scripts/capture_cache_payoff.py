"""Capture the cache-payoff animation from the tutorial notebook.

A real server, the notebook ``strata tutorial`` opens, its environment warmed the same way, and
real browsers clicking and typing in it. Every beat waits for the cells to reach the states it
names, so a cache that stopped working fails the capture instead of producing a convincing GIF.

    uv run python scripts/capture_cache_payoff.py

Writes ``docs/assets/cache-payoff-{light,dark}.gif``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from assemble_gif import assemble  # noqa: E402
from capture_docs_shots import _free_port, check_bundle_is_current, serve  # noqa: E402

# Opening a sample syncs and warms its environment, as `strata tutorial` waits for it to.
OPEN_TIMEOUT_SECONDS = 600


def _open_tutorial(base_url: str) -> str:
    request = urllib.request.Request(
        f"{base_url}/v1/notebooks/samples/caching/open",
        method="POST",
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=OPEN_TIMEOUT_SECONDS) as response:
        return str(json.load(response)["session_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="/tmp/strata-cache-payoff")
    args = parser.parse_args()
    work_dir = Path(args.work_dir)

    check_bundle_is_current()
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)

    port = _free_port()
    server = serve(work_dir, port)
    base_url = f"http://127.0.0.1:{port}"
    frame_dir = work_dir / "frames"
    try:
        subprocess.run(
            [
                "node",
                "scripts/capture-web-cache-payoff.mjs",
                "--base-url",
                base_url,
                "--session",
                _open_tutorial(base_url),
                "--out",
                str(frame_dir),
            ],
            cwd=REPO_ROOT / "frontend",
            check=True,
        )
    finally:
        server.terminate()
        server.wait(timeout=30)

    for theme in ("light", "dark"):
        out = assemble(frame_dir, f"cache-payoff-{theme}")
        print(f"wrote {out.relative_to(REPO_ROOT)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
