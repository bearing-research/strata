"""Capture a real agent-drives / TUI-observes session.

Everything else in `capture_docs_shots.py` feeds the TUI *canned* frames: no
server, no network, deterministic by construction. That is right for a layout
diagram and wrong for this one, because the claim being illustrated is that a
notebook driven from one terminal shows up live in a viewer in another. Canned
frames cannot demonstrate that; they would show it whether or not it worked.

So this drives the real thing. A real server, a real notebook, the real
`strata` CLI issuing the commands a coding agent issues, and a real TUI client
on a real WebSocket rendering whatever arrives. The frames are screenshots of
that session.

    uv run python scripts/capture_agent_drive.py

Doubles as an end-to-end check of the live-mirror path (#357): every beat
asserts the TUI actually received the change, so a regression that stopped the
broadcast would fail the capture rather than quietly produce a still-looking
animation.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from capture_docs_shots import ASSETS, TUI_SIZE, _free_port, _strata, serve  # noqa: E402

# What the "agent" types. Deliberately the CLI a coding agent actually uses -
# not REST calls dressed up - so the capture exercises the same path the
# scratchpad skill and the MCP server take.
LOADER = "import time\ntime.sleep(1)\nrows = [1, 2, 3, 4, 5]\nsum(rows)\n"
DERIVED = "doubled = [r * 2 for r in rows]\ndoubled\n"


def _drive(server: str, session: str, *args: str) -> None:
    """Run one `strata` command against the *live session*, as a subprocess.

    ``--server``/``--session``, not a notebook path. The local backend edits
    files on disk and the server never learns of it, so a watcher sees nothing
    until its next resync — the first version of this capture drove that way
    and the TUI stayed empty, which is the whole distinction being illustrated.
    """
    _strata(*args, "--server", server, "--session", session)


async def _run(work_dir: Path) -> list[Path]:
    from strata.notebook.tui.app import NotebookTUI
    from strata.notebook.tui.client import TuiClient

    # A clean slate, every run. The beats assert on exact cell counts, so a
    # notebook left behind by an earlier run makes the first assertion fail
    # against a graph that was never empty — which is exactly how this capture
    # first misread a working live mirror as a broken one.
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

                The assertion is on the viewer's state, never on the CLI's exit
                code: a command that succeeded while the broadcast was lost is
                exactly the failure worth catching, and it is what the first
                version of this capture hit by driving the local backend.

                Against ``app.vm`` rather than the rendered text, because the
                cell list shows a *source preview* — asserting on a fragment
                that happens to fall outside it fails while the mirror is
                working perfectly, which is a worse outcome than no check.
                """
                for _ in range(tries):
                    await pilot.pause()
                    await asyncio.sleep(0.25)
                    if not ready():
                        continue
                    # Let the trailing frames land. Status flips a beat before
                    # the output arrives, and snapshotting on the flip
                    # photographs a finished cell reporting "(no output)".
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

            # Cell ids from the viewer's own model: it is the state this
            # capture already trusts, and it needs neither a disk read nor
            # another subprocess to agree with what is on screen.
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
