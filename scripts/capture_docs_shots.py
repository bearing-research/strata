#!/usr/bin/env python
"""Regenerate the screenshots the docs and README embed.

Screenshots rot silently: a shot of last release's layout still *looks*
authoritative. So none of them are hand-made — every image under
``docs/assets/`` comes out of this script, and refreshing all of them after a
UI change is one command:

    uv run python scripts/capture_docs_shots.py

Two very different capture paths:

* **TUI** shots are SVG, captured in-process with Textual's pilot driver. No
  server, no network, no terminal: canned ``notebook_state`` frames are fed
  straight into ``NotebookTUI._dispatch``, exactly as ``tests/notebook/
  test_tui_app.py`` does. Fully deterministic. SVG keeps the text real, so the
  shot stays crisp at any zoom and reads on both docs themes.

* **Web** shots are PNG, captured by ``frontend/scripts/capture-docs-shots.mjs``
  driving Playwright against a real server. This script builds the fixture
  notebook, runs it headlessly, serves it, and hands the session id to node.

The web fixture's cell sources are extracted from
``docs/getting-started/notebook.md`` itself, so a screenshot cannot show
different code than the page it illustrates.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "docs" / "assets"
QUICKSTART_DOC = REPO_ROOT / "docs" / "getting-started" / "notebook.md"


# --------------------------------------------------------------------------
# TUI shots (SVG, no server)
# --------------------------------------------------------------------------

# Terminal geometry for the TUI captures. Wide enough that the cell list's
# "time" column survives (the label column carries a source preview, so a
# narrow terminal pushes timing off the edge), short enough that the SVG does
# not dwarf the prose around it.
TUI_SIZE = (132, 28)

# The loader's tabular output, in the shape the server sends: column names
# plus a row preview. The TUI renders that as a real grid (ellipsis-truncated,
# no wrapping), which is what a reader should see in the output pane.
IRIS_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "target",
    "species",
]
IRIS_ROWS = [
    [5.1, 3.5, 1.4, 0.2, 0, "setosa"],
    [4.9, 3.0, 1.4, 0.2, 0, "setosa"],
    [4.7, 3.2, 1.3, 0.2, 0, "setosa"],
    [4.6, 3.1, 1.5, 0.2, 0, "setosa"],
    [5.0, 3.6, 1.4, 0.2, 0, "setosa"],
]

QUICKSTART_CELLS = [
    {
        "id": "a1b2c3d4",
        "name": "load",
        "status": "ready",
        "source": (
            "import time\n"
            "import pandas as pd\n"
            "from sklearn.datasets import load_iris\n"
            "\n"
            "time.sleep(2)  # simulate the latency of a real fetch\n"
            "iris = load_iris(as_frame=True)\n"
            "df = iris.frame.copy()\n"
            'df["species"] = pd.Categorical.from_codes(df["target"], iris.target_names)\n'
            "feature_names = iris.feature_names\n"
            "df.head()\n"
        ),
        "display_outputs": [
            {
                "content_type": "arrow/ipc",
                "columns": IRIS_COLUMNS,
                "preview": IRIS_ROWS,
                "rows": 150,
            }
        ],
    },
    {
        "id": "e5f6a7b8",
        "name": "summarize",
        "status": "ready",
        "source": (
            'stats = df.groupby("species", observed=True)[feature_names].mean().round(2)\nstats\n'
        ),
    },
    {
        "id": "c9d0e1f2",
        "name": "plot",
        "status": "ready",
        "source": (
            "import matplotlib.pyplot as plt\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(6, 4))\n"
            'for species, group in df.groupby("species", observed=True):\n'
            "    ax.scatter(\n"
            '        group["sepal length (cm)"],\n'
            '        group["petal length (cm)"],\n'
            "        label=str(species),\n"
            "        alpha=0.7,\n"
            "    )\n"
            "fig\n"
        ),
    },
]


def _frame(msg_type: str, payload: dict) -> str:
    return json.dumps({"type": msg_type, "seq": 0, "ts": "t", "payload": payload})


def _timing(cell_id: str, *, duration_ms: int, cache_hit: bool = False) -> str:
    return _frame(
        "cell_output",
        {"cell_id": cell_id, "duration_ms": duration_ms, "cache_hit": cache_hit, "outputs": []},
    )


async def _capture_tui(name: str, title: str, script) -> Path:
    """Run ``script`` against a pilot-driven TUI and write its SVG."""
    from strata.notebook.tui.app import NotebookTUI
    from strata.notebook.tui.client import TuiClient

    async def _noop(self) -> None:  # never touch the network
        return None

    original_bootstrap = NotebookTUI._bootstrap
    NotebookTUI._bootstrap = _noop  # type: ignore[method-assign]
    try:
        app = NotebookTUI(client=TuiClient("http://localhost:8765"), session_id="docs")
        async with app.run_test(size=TUI_SIZE) as pilot:
            app._set_connection("connected")
            await script(app, pilot)
            # The label column carries a source preview, so at the default
            # split the timing column is pushed off the edge. Nudge the
            # boundary (the app's own ctrl+right binding) so shots show it.
            for _ in range(3):
                await pilot.press("ctrl+right")
            await pilot.pause()
            svg = app.export_screenshot(title=title)
    finally:
        NotebookTUI._bootstrap = original_bootstrap  # type: ignore[method-assign]

    out = ASSETS / f"{name}.svg"
    out.write_text(svg, encoding="utf-8")
    return out


async def _tui_layout(app, pilot) -> None:
    """A settled notebook: three cells, the loader selected, its output below."""
    app._dispatch(_frame("notebook_state", {"name": "iris", "cells": QUICKSTART_CELLS}))
    await pilot.pause()
    app._dispatch(_timing("a1b2c3d4", duration_ms=2043))
    app._dispatch(_timing("e5f6a7b8", duration_ms=38, cache_hit=True))
    app._dispatch(_timing("c9d0e1f2", duration_ms=412))
    await pilot.pause()
    app._select_cell("a1b2c3d4")


async def _tui_agent_running(app, pilot) -> None:
    """Mid-run: the agent has edited the plot cell and it is executing now."""
    cells = [dict(c) for c in QUICKSTART_CELLS]
    cells[2] = {**cells[2], "status": "running"}
    app._dispatch(_frame("notebook_state", {"name": "scratch", "cells": cells}))
    await pilot.pause()
    app._dispatch(_timing("a1b2c3d4", duration_ms=2043))
    app._dispatch(_timing("e5f6a7b8", duration_ms=38, cache_hit=True))
    app._dispatch(_frame("cell_status", {"cell_id": "c9d0e1f2", "status": "running"}))
    for line in (
        "loading cached df from nb_iris_cell_a1b2c3d4_var_df\n",
        "rendering scatter for 3 species\n",
    ):
        app._dispatch(
            _frame("cell_console", {"cell_id": "c9d0e1f2", "stream": "stdout", "text": line})
        )
    await pilot.pause()
    await pilot.press("5")  # Console tab


TUI_SHOTS = {
    "tui-layout": ("strata watch", _tui_layout),
    "tui-agent-running": ("strata agent", _tui_agent_running),
}


def capture_tui() -> list[Path]:
    written = []
    for name, (title, script) in TUI_SHOTS.items():
        written.append(asyncio.run(_capture_tui(name, title, script)))
    return written


# --------------------------------------------------------------------------
# Web shots (PNG, real server + Playwright)
# --------------------------------------------------------------------------


def extract_quickstart_cells() -> list[str]:
    """Pull the quickstart's three python blocks out of the doc that shows them.

    Binding the fixture to the doc's own source is the point: it makes it
    impossible for the screenshot to show code the page doesn't.
    """
    text = QUICKSTART_DOC.read_text(encoding="utf-8")
    start = text.index("## 3. Walk through a pipeline")
    end = text.index("## 4. Re-run for cache hits")
    blocks = re.findall(r"```python\n(.*?)```", text[start:end], re.DOTALL)
    if len(blocks) != 3:
        raise SystemExit(
            f"expected 3 python blocks in the quickstart pipeline, found {len(blocks)}. "
            "The doc was restructured; update extract_quickstart_cells()."
        )
    return blocks


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _strata(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(
        [sys.executable, "-m", "strata.cli", *args],
        cwd=cwd or REPO_ROOT,
        check=True,
    )


# The registry walkthrough's fixture: one cell that trains a model and
# publishes it under a name. Kept here rather than in ``examples/`` because it
# exists only to give the registry surfaces something to show.
REGISTRY_CELL = """\
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([2.0, 4.1, 5.9, 8.2])
model = LinearRegression().fit(X, y)

art = strata.put(
    inputs=[],
    transform={"ref": "train-tip-model@v1"},
    data={"coef": model.coef_.tolist(), "intercept": [float(model.intercept_)]},
    name="taxi/tip-model",
)
print(art.uri)
"""


def _scaffold(root: Path, name: str, deps: tuple[str, ...], cells: list[str]) -> Path:
    nb = root / name
    if nb.exists():
        shutil.rmtree(nb)
    root.mkdir(parents=True, exist_ok=True)

    _strata("new", name, "--parent", str(root), "--no-env")
    for dep in deps:
        _strata("dep", "add", str(nb), dep)
    for source in cells:
        _strata("cell", "add", str(nb), "-c", source)
    return nb


def build_fixtures(root: Path) -> tuple[Path, Path]:
    """Scaffold the two fixture notebooks the web shots photograph.

    The quickstart notebook is run here so its cells carry real outputs. The
    registry notebook is not: its cell publishes through the ambient ``strata``
    client, so it has to execute against the running server, which the browser
    script does over REST.
    """
    iris = _scaffold(
        root, "iris", ("pandas", "scikit-learn", "matplotlib"), extract_quickstart_cells()
    )
    _strata("run", str(iris))
    registry = _scaffold(root, "registry", ("scikit-learn",), [REGISTRY_CELL])
    return iris, registry


def serve(storage_dir: Path, port: int) -> subprocess.Popen:
    env = {
        **os.environ,
        "STRATA_NOTEBOOK_STORAGE_DIR": str(storage_dir),
        "STRATA_DEPLOYMENT_MODE": "personal",
        "STRATA_PORT": str(port),
        # Isolate the artifact store: the default (~/.strata/artifacts) carries
        # whatever the developer's own notebooks published, and those names show
        # up in the Registry tab shot.
        "STRATA_ARTIFACT_DIR": str(storage_dir / ".artifacts"),
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "strata"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit("strata server exited before it came up")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return proc
        except OSError:
            time.sleep(0.25)
    proc.terminate()
    raise SystemExit(f"strata server did not listen on {port} within 60s")


def capture_web(work_dir: Path) -> None:
    iris, registry = build_fixtures(work_dir)
    port = _free_port()
    proc = serve(work_dir, port)
    try:
        subprocess.run(
            [
                "node",
                "scripts/capture-docs-shots.mjs",
                "--base-url",
                f"http://127.0.0.1:{port}",
                "--iris-path",
                str(iris),
                "--registry-path",
                str(registry),
                "--out",
                str(ASSETS),
            ],
            cwd=REPO_ROOT / "frontend",
            check=True,
        )
    finally:
        proc.terminate()
        proc.wait(timeout=30)


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("all", "tui", "web"), default="all")
    parser.add_argument(
        "--work-dir",
        default="/tmp/strata-docs-shots",
        help="Scratch directory for the web fixture notebook",
    )
    args = parser.parse_args()

    ASSETS.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if args.only in ("all", "tui"):
        written += capture_tui()
    if args.only in ("all", "web"):
        capture_web(Path(args.work_dir))
        written += sorted(ASSETS.glob("*.png"))

    for path in written:
        print(f"{path.relative_to(REPO_ROOT)}  {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
