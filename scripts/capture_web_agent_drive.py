"""Capture the README's live web animation: the CLI drives, a browser watches.

The web counterpart of ``capture_agent_drive.py``: a real server, a real
notebook driven with the ``strata`` CLI over ``--server``/``--session`` (the
path a coding agent takes), and real browsers on the same session. Every beat
waits for the browsers to show its change, so a broken broadcast fails the
capture instead of producing a convincing GIF.

    uv run python scripts/capture_web_agent_drive.py

Writes ``docs/assets/web-agent-live-{light,dark}.gif``.
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
from capture_docs_shots import _free_port, _strata, check_bundle_is_current, serve  # noqa: E402


def _open_session(base_url: str, notebook: Path) -> str:
    request = urllib.request.Request(
        f"{base_url}/v1/notebooks/open",
        data=json.dumps({"path": str(notebook)}).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return str(json.load(response)["session_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="/tmp/strata-web-agent-drive")
    args = parser.parse_args()
    work_dir = Path(args.work_dir)

    check_bundle_is_current()
    # The beats count cells, so a leftover notebook would fail them.
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    _strata("new", "live", "--parent", str(work_dir), "--no-env")

    port = _free_port()
    server = serve(work_dir, port)
    base_url = f"http://127.0.0.1:{port}"
    frame_dir = work_dir / "frames"
    try:
        subprocess.run(
            [
                "node",
                "scripts/capture-web-agent-live.mjs",
                "--base-url",
                base_url,
                "--session",
                _open_session(base_url, work_dir / "live"),
                "--python",
                sys.executable,
                "--repo-root",
                str(REPO_ROOT),
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
        out = assemble(frame_dir, f"web-agent-live-{theme}")
        print(f"wrote {out.relative_to(REPO_ROOT)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
