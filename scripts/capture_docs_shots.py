#!/usr/bin/env python
"""Regenerate the screenshots the docs and README embed.

Every image under ``docs/assets/`` comes from this script, so none can silently
show an old layout. Refresh them all after a UI change with:

    uv run python scripts/capture_docs_shots.py

* **TUI** shots are SVG from Textual's pilot driver, fed canned
  ``notebook_state`` frames (as ``tests/notebook/test_tui_app.py`` does): no
  server, fully deterministic.
* **Web** shots are PNG from ``frontend/scripts/capture-docs-shots.mjs``
  driving Playwright against a real server this script sets up.

The web fixture's cells are extracted from ``docs/getting-started/notebook.md``,
so a shot cannot show different code than the page it illustrates.
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
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "docs" / "assets"
QUICKSTART_DOC = REPO_ROOT / "docs" / "getting-started" / "notebook.md"


# --------------------------------------------------------------------------
# TUI shots (SVG, no server)
# --------------------------------------------------------------------------

# Wide enough to keep the cell list's "time" column on screen, short enough
# not to dwarf the surrounding prose.
TUI_SIZE = (132, 28)

# The loader's tabular output in the shape the server sends, so the TUI renders
# a real grid.
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
            # At the default split the source preview pushes timing off screen.
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


async def _capture_tui_frames(title: str, script) -> list[str]:
    """Run ``script`` against a pilot-driven TUI, returning one SVG per beat.

    Canned frames, as for the stills. ``script`` calls ``snap`` at each beat,
    so the storyboard is explicit rather than inferred from timing.
    """
    from strata.notebook.tui.app import NotebookTUI
    from strata.notebook.tui.client import TuiClient

    async def _noop(self) -> None:
        return None

    frames: list[str] = []
    original_bootstrap = NotebookTUI._bootstrap
    NotebookTUI._bootstrap = _noop  # type: ignore[method-assign]
    try:
        app = NotebookTUI(client=TuiClient("http://localhost:8765"), session_id="docs")
        async with app.run_test(size=TUI_SIZE) as pilot:
            app._set_connection("connected")
            for _ in range(3):
                await pilot.press("ctrl+right")

            async def snap() -> None:
                await pilot.pause()
                frames.append(app.export_screenshot(title=title))

            await script(app, pilot, snap)
    finally:
        NotebookTUI._bootstrap = original_bootstrap  # type: ignore[method-assign]
    return frames


async def _tui_cache_payoff(app, pilot, snap) -> None:
    """The README animation: a downstream edit, and the slow upstream never reruns.

    The quickstart's ``load`` sleeps two seconds, so a rerun would be noticeable.
    """
    ready = [dict(c) for c in QUICKSTART_CELLS]

    # 1 — settled. Everything green, load cost 2s.
    app._dispatch(_frame("notebook_state", {"name": "iris", "cells": ready}))
    await pilot.pause()
    app._dispatch(_timing("a1b2c3d4", duration_ms=2043))
    app._dispatch(_timing("e5f6a7b8", duration_ms=38))
    app._dispatch(_timing("c9d0e1f2", duration_ms=380))
    app._select_cell("c9d0e1f2")
    await snap()

    # 2 — the plot cell is edited, so it and only it goes stale.
    app._dispatch(_frame("cell_status", {"cell_id": "c9d0e1f2", "status": "idle"}))
    await snap()

    # 3 — the run starts. Upstream resolves from cache without executing.
    app._dispatch(_timing("a1b2c3d4", duration_ms=1, cache_hit=True))
    app._dispatch(_timing("e5f6a7b8", duration_ms=1, cache_hit=True))
    app._dispatch(_frame("cell_status", {"cell_id": "c9d0e1f2", "status": "running"}))
    # Show Console now, or the next beat's console lines would not be visible.
    await pilot.press("5")
    await snap()

    # 4 — only the edited cell actually computes, and it says so.
    for line in (
        "loading cached df from nb_iris_cell_a1b2c3d4_var_df\n",
        "rendering scatter for 3 species\n",
    ):
        app._dispatch(
            _frame("cell_console", {"cell_id": "c9d0e1f2", "stream": "stdout", "text": line})
        )
    await snap()

    # 5 — done. load still reads cached; the two seconds were not paid again.
    app._dispatch(_frame("cell_status", {"cell_id": "c9d0e1f2", "status": "ready"}))
    app._dispatch(_timing("c9d0e1f2", duration_ms=412))
    await snap()


TUI_ANIMATIONS = {
    "tui-cache-payoff": ("strata watch", _tui_cache_payoff),
}


def capture_tui_animation_frames(out_dir: Path) -> dict[str, list[Path]]:
    """Write each animation's SVG beats to *out_dir*; rasterising is a separate step."""
    written: dict[str, list[Path]] = {}
    for name, (title, script) in TUI_ANIMATIONS.items():
        frames = asyncio.run(_capture_tui_frames(title, script))
        paths = []
        for i, svg in enumerate(frames):
            f = out_dir / f"{name}-{i:02d}.svg"
            f.write_text(svg, encoding="utf-8")
            paths.append(f)
        written[name] = paths
    return written


TUI_SHOTS = {
    "tui-layout": ("strata watch", _tui_layout),
    "tui-agent-running": ("strata agent", _tui_agent_running),
}


def capture_animations(frame_dir: Path) -> list[Path]:
    """SVG beats → PNG (node/Playwright) → GIF (Pillow).

    Nothing here renders SVG from Python; Playwright (a frontend dependency)
    renders it as a reader's browser would.
    """
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("*.png"):
        old.unlink()  # stale frames would be packed into the next GIF
    written = capture_tui_animation_frames(frame_dir)

    subprocess.run(
        ["node", "scripts/rasterize-svg.mjs", "--in", str(frame_dir)],
        cwd=REPO_ROOT / "frontend",
        check=True,
    )
    out = []
    for name in written:
        subprocess.run(
            [sys.executable, "scripts/assemble_gif.py", "--in", str(frame_dir), "--name", name],
            cwd=REPO_ROOT,
            check=True,
        )
        out.append(ASSETS / f"{name}.gif")
    return out


def capture_tui() -> list[Path]:
    written = []
    for name, (title, script) in TUI_SHOTS.items():
        written.append(asyncio.run(_capture_tui(name, title, script)))
    return written


# --------------------------------------------------------------------------
# Web shots (PNG, real server + Playwright)
# --------------------------------------------------------------------------


def _anchor(text: str, heading: str) -> int:
    """Locate a heading the fixture is keyed to, or say which one moved."""
    index = text.find(heading)
    if index < 0:
        raise SystemExit(
            f"{QUICKSTART_DOC.name} no longer contains the heading {heading!r}. "
            "The fixture is extracted from that section; update "
            "extract_quickstart_cells() to match the new structure."
        )
    return index


def extract_quickstart_cells() -> list[str]:
    """Pull the quickstart's three python blocks out of the doc that shows them.

    The screenshot then cannot show code the page does not.
    """
    text = QUICKSTART_DOC.read_text(encoding="utf-8")
    start = _anchor(text, "## 3. Walk through a pipeline")
    end = _anchor(text, "## 4. Re-run for cache hits")
    blocks = re.findall(r"```python\n(.*?)```", text[start:end], re.DOTALL)
    if len(blocks) != 3:
        raise SystemExit(
            f"expected 3 python blocks in the quickstart pipeline, found {len(blocks)}. "
            "The doc was restructured; update extract_quickstart_cells()."
        )
    return blocks


def serving_bundle() -> Path | None:
    """The built frontend ``python -m strata`` will serve (``_mount_frontend``'s order)."""
    for candidate in (
        REPO_ROOT / "src" / "strata" / "_frontend",
        REPO_ROOT / "frontend" / "dist",
    ):
        if (candidate / "index.html").exists():
            return candidate
    return None


def _newest_mtime(root: Path, pattern: str = "**/*") -> float:
    return max((f.stat().st_mtime for f in root.glob(pattern) if f.is_file()), default=0.0)


def check_bundle_is_current() -> Path:
    """Refuse to photograph a stale bundle.

    Shots use the built bundle (what users install), and the gitignored
    ``src/strata/_frontend/`` wins over ``frontend/dist/`` but is only refreshed
    by a manual copy, so a stale one would silently re-shoot the old UI.
    """
    bundle = serving_bundle()
    if bundle is None:
        raise SystemExit(
            "No built frontend found. Build it first:\n"
            "  cd frontend && npm run build && cp -r dist/ ../src/strata/_frontend/"
        )
    if _newest_mtime(REPO_ROOT / "frontend" / "src") > _newest_mtime(bundle):
        raise SystemExit(
            f"{bundle.relative_to(REPO_ROOT)} is older than frontend/src — the shots "
            "would show the previous UI. Rebuild it first:\n"
            "  cd frontend && npm run build && cp -r dist/ ../src/strata/_frontend/"
        )
    return bundle


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


# The registry walkthrough shows "model <- features <- scan <- table @ snapshot",
# so the fixture builds that chain for real; one put would show a one-row lineage.
REGISTRY_CELLS = [
    """\
raw = strata.materialize(
    inputs=[TABLE_URI],
    transform={"ref": "scan@v1", "params": {}},
    name="taxi/raw-trips",
)
print(raw.uri)
""",
    """\
features = strata.put(
    inputs=[raw.uri],
    transform={"ref": "tip-features@v1"},
    data={"fare": [12.5, 8.0, 22.25], "distance": [3.1, 1.4, 7.8]},
    name="taxi/features",
)
print(features.uri)
""",
    """\
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[3.1], [1.4], [7.8]])
y = np.array([2.4, 1.1, 4.6])
model = LinearRegression().fit(X, y)

art = strata.put(
    inputs=[features.uri],
    transform={"ref": "train-tip-model@v1"},
    data={"coef": model.coef_.tolist(), "intercept": [float(model.intercept_)]},
    name="taxi/tip-model",
)
print(art.uri)
""",
]


def build_warehouse(root: Path) -> str:
    """A one-table Iceberg warehouse, so the lineage chain has a real root.

    Mirrors the ``temp_warehouse`` test fixture.
    """
    import pyarrow as pa
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import DoubleType, LongType, NestedField

    warehouse = root / "warehouse"
    warehouse.mkdir(parents=True, exist_ok=True)
    catalog = SqlCatalog(
        "strata",
        uri=f"sqlite:///{(warehouse / 'catalog.db').as_posix()}",
        warehouse=warehouse.as_uri(),
    )
    catalog.create_namespace("taxi")
    table = catalog.create_table(
        "taxi.trips",
        Schema(
            NestedField(1, "trip_id", LongType(), required=False),
            NestedField(2, "fare", DoubleType(), required=False),
            NestedField(3, "distance", DoubleType(), required=False),
        ),
    )
    table.append(
        pa.table(
            {
                "trip_id": pa.array(range(200), type=pa.int64()),
                "fare": pa.array([5.0 + i * 0.25 for i in range(200)], type=pa.float64()),
                "distance": pa.array([1.0 + i * 0.05 for i in range(200)], type=pa.float64()),
            }
        )
    )
    return f"{warehouse.as_uri()}#taxi.trips"


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


# Real assertions for the quickstart's ``summarize`` cell; a reader may copy them.
# Only two: ``.tests-panel`` caps at 420px, and more would clip the editor and
# push results out of view, which photographs as a rendering bug.
SUMMARIZE_TESTS = """\
def test_one_row_per_species(cell):
    assert len(cell.stats) == 3


def test_setosa_has_the_shortest_petals(cell):
    assert cell.stats["petal length (cm)"].idxmin() == "setosa"
"""


def _cell_id(notebook: Path, snippet: str) -> str:
    """Resolve a cell id by a distinctive fragment of its source.

    Ids are backend-generated, the quickstart cells carry no name annotations
    (adding them would show in the screenshot), and matching by position would
    silently pick the wrong cell if the quickstart gained a step.
    """
    out = subprocess.run(
        [sys.executable, "-m", "strata.cli", "cell", "list", str(notebook), "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    matches = [c for c in json.loads(out.stdout) if snippet in (c.get("source") or "")]
    if len(matches) != 1:
        raise SystemExit(
            f"{len(matches)} cells in {notebook} contain {snippet!r}; expected exactly one. "
            "The quickstart source changed — pick a fragment that is still unique."
        )
    return str(matches[0]["id"])


def publish_figure(notebook: Path, storage_dir: Path) -> str:
    """Publish the quickstart's plot and return its token.

    Publishes for real so the page shows what the code produces.
    ``STRATA_ARTIFACT_DIR`` is the store ``serve()`` reads, so publishing copies
    the plot out of the notebook's ``.strata/artifacts`` and the link resolves.
    """
    plot_cell = _cell_id(notebook, "fig, ax = plt.subplots")
    notebook_id = tomllib.loads((notebook / "notebook.toml").read_text(encoding="utf-8"))[
        "notebook_id"
    ]
    artifact_id = f"nb_{notebook_id}_cell_{plot_cell}_var___display__0"

    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "strata.cli",
            "artifact",
            "publish",
            artifact_id,
            "--artifact-dir",
            str(notebook / ".strata" / "artifacts"),
            "--title",
            "Iris feature distributions",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
        env={**os.environ, "STRATA_ARTIFACT_DIR": str(storage_dir / ".artifacts")},
    )
    return str(json.loads(out.stdout)["token"])


def build_fixtures(root: Path) -> tuple[Path, Path]:
    """Scaffold the two fixture notebooks the web shots photograph.

    The quickstart notebook is run here. The registry notebook is not: it
    publishes through the ambient ``strata`` client, so the browser script runs
    it against the live server.
    """
    iris = _scaffold(
        root, "iris", ("pandas", "scikit-learn", "matplotlib"), extract_quickstart_cells()
    )
    _strata("run", str(iris))

    # Run real tests so the Tests panel shows a result. Results persist to
    # runtime.json, so the browser finds them on open.
    tests = root / "summarize.test.py"
    tests.write_text(SUMMARIZE_TESTS, encoding="utf-8")
    _strata("cell", "test", str(iris), _cell_id(iris, "stats = df.groupby"), "--file", str(tests))

    # Bind the table URI as a literal: an env var would show in the screenshot
    # as unexplained indirection.
    table_uri = build_warehouse(root)
    cells = [REGISTRY_CELLS[0].replace("TABLE_URI", repr(table_uri)), *REGISTRY_CELLS[1:]]
    registry = _scaffold(root, "registry", ("scikit-learn",), cells)
    return iris, registry


def serve(storage_dir: Path, port: int) -> subprocess.Popen:
    env = {
        **os.environ,
        "STRATA_NOTEBOOK_STORAGE_DIR": str(storage_dir),
        "STRATA_DEPLOYMENT_MODE": "personal",
        "STRATA_PORT": str(port),
        # Isolated so the developer's own published names stay out of the shot.
        "STRATA_ARTIFACT_DIR": str(storage_dir / ".artifacts"),
    }
    # Keep the log: an inherited STRATA_AUTH_MODE or STRATA_MULTI_TENANT_ENABLED
    # fails validate_mode_coherence against the forced personal mode.
    log = storage_dir / "server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("wb")
    proc = subprocess.Popen(
        [sys.executable, "-m", "strata"],
        cwd=REPO_ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(
                f"strata server exited before it came up:\n{log.read_text(errors='replace')}"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return proc
        except OSError:
            time.sleep(0.25)
    proc.terminate()
    raise SystemExit(
        f"strata server did not listen on {port} within 60s:\n{log.read_text(errors='replace')}"
    )


def capture_web(work_dir: Path) -> None:
    check_bundle_is_current()
    iris, registry = build_fixtures(work_dir)
    token = publish_figure(iris, work_dir)
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
                "--publication-token",
                token,
                "--out",
                str(ASSETS),
            ],
            cwd=REPO_ROOT / "frontend",
            check=True,
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            # Never let cleanup mask the capture error or leave the port held.
            proc.kill()
            proc.wait(timeout=10)


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("all", "tui", "web", "anim"), default="all")
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
    if args.only in ("all", "anim"):
        written += capture_animations(Path(args.work_dir) / "frames")

    for path in written:
        print(f"{path.relative_to(REPO_ROOT)}  {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
