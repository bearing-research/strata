"""Capture a real agent-drives / TUI-observes session.

Unlike the canned frames in `capture_docs_shots.py`, this drives a real server
and notebook with the `strata` CLI and screenshots a real TUI client, because
the claim shown (a notebook driven from one terminal appears live in another)
is one canned frames would show whether or not it worked.

    uv run python scripts/capture_agent_drive.py

Doubles as an end-to-end check of the live mirror: every beat asserts the TUI
received the change, so a broken broadcast fails the capture.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from capture_docs_shots import ASSETS, TUI_SIZE, _free_port, _strata, serve  # noqa: E402

# What the "agent" types, through the same CLI a coding agent uses.
LOADER = "import time\ntime.sleep(1)\nrows = [1, 2, 3, 4, 5]\nsum(rows)\n"
DERIVED = "doubled = [r * 2 for r in rows]\ndoubled\n"


def _drive(server: str, session: str, *args: str) -> None:
    """Run one `strata` command against the *live session*, as a subprocess.

    Not a notebook path: the local backend edits files the server never hears
    about, so the TUI would see nothing until its next resync.
    """
    _strata(*args, "--server", server, "--session", session)


async def _run(work_dir: Path) -> list[Path]:
    from strata.notebook.tui.app import NotebookTUI
    from strata.notebook.tui.client import TuiClient

    # The beats assert exact cell counts, so a leftover notebook would fail them.
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    port = _free_port()
    server = serve(work_dir, port)
    frames: list[str] = []

    try:
        notebook = work_dir / "live"
        # Scaffolding, not driving: this runs before a session exists.
        _strata("new", "live", "--parent", str(work_dir), "--no-env")

        app = NotebookTUI(
            client=TuiClient(f"http://127.0.0.1:{port}"),
            notebook_path=str(notebook),
        )
        async with app.run_test(size=TUI_SIZE) as pilot:
            # Let the real bootstrap open the session and connect the socket.
            for _ in range(40):
                await pilot.pause()
                await asyncio.sleep(0.25)
                if app._session_id:
                    break
            if not app._session_id:
                raise SystemExit("the TUI never connected; nothing to capture")
            for _ in range(3):
                await pilot.press("ctrl+right")

            async def settle(what: str, ready, tries: int = 60) -> None:
                """Wait until the TUI's own model satisfies *ready*, then snapshot.

                Checks the viewer's state, not the CLI's exit code, since a
                lost broadcast is the failure worth catching. Uses ``app.vm``
                rather than rendered text, which shows only a source preview.
                """
                for _ in range(tries):
                    await pilot.pause()
                    await asyncio.sleep(0.25)
                    if not ready():
                        continue
                    # Status flips a beat before the output arrives; wait for it.
                    for _ in range(4):
                        await pilot.pause()
                        await asyncio.sleep(0.25)
                    frames.append(app.export_screenshot(title="strata watch"))
                    return
                raise SystemExit(
                    f"the TUI never reached {what!r} — the live mirror is broken, "
                    "or the beat needs longer than it was given"
                )

            def cells_seen() -> list:
                return list(app.vm.cells.values())

            base_url = f"http://127.0.0.1:{port}"
            sid = app._session_id
            frames.append(app.export_screenshot(title="strata watch"))  # empty notebook

            await asyncio.to_thread(_drive, base_url, sid, "cell", "add", "-c", LOADER)
            await settle("the first cell appearing", lambda: len(cells_seen()) == 1)

            # Cell ids from the viewer's model, which matches what is on screen.
            await asyncio.to_thread(_drive, base_url, sid, "cell", "run", app.vm.cell_order[0])
            await settle(
                "the first cell finishing",
                lambda: any(c.status == "ready" for c in cells_seen()),
            )

            await asyncio.to_thread(_drive, base_url, sid, "cell", "add", "-c", DERIVED)
            await settle("the second cell appearing", lambda: len(cells_seen()) == 2)

            await asyncio.to_thread(_drive, base_url, sid, "cell", "run", app.vm.cell_order[1])
            await settle(
                "both cells finishing",
                lambda: sum(c.status == "ready" for c in cells_seen()) == 2,
            )
    finally:
        server.terminate()
        server.wait(timeout=10)

    out_dir = work_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, svg in enumerate(frames):
        f = out_dir / f"tui-agent-live-{i:02d}.svg"
        f.write_text(svg, encoding="utf-8")
        written.append(f)
    return written


def main() -> None:
    work = Path("/tmp/strata-agent-drive")
    written = asyncio.run(_run(work))
    print(f"{len(written)} frames in {work / 'frames'}")
    for f in written:
        print(f"  {f.name}")
    print("\nRasterise + assemble:")
    print(f"  cd frontend && node scripts/rasterize-svg.mjs --in {work / 'frames'}")
    print(f"  uv run python scripts/assemble_gif.py --in {work / 'frames'} --name tui-agent-live")
    print(f"\nASSETS = {ASSETS}")


if __name__ == "__main__":
    main()
