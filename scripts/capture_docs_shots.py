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
import contextlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "docs" / "assets"
QUICKSTART_DOC = REPO_ROOT / "docs" / "getting-started" / "notebook.md"


# --------------------------------------------------------------------------
# TUI shots (SVG, no server)
# --------------------------------------------------------------------------

# A common terminal size, short enough not to dwarf the surrounding prose.
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


# The lake notebook shows what a run reads and writes: the weight badge and the
# read dialog need a table big enough to cross the threshold below, and the
# folder listing needs files of real size.
LAKE_CELLS = [
    """\
# @name rides
# @table rides lake:taxi.rides
scan = strata.materialize(
    inputs=[rides],
    transform={"executor": "scan@v1", "params": {"snapshot_id": rides_snapshot}},
)
df = scan.to_pandas()
print(f"{len(df):,} rides at snapshot {rides_snapshot}")
""",
    """\
# @name train
# @output_dir ckpt
import json
import numpy as np

fare_per_km = float((df["fare"] / df["distance"]).mean())
(ckpt / "metrics.json").write_text(json.dumps({"fare_per_km": round(fare_per_km, 3)}))
rng = np.random.default_rng(0)
for epoch in range(3):
    np.save(ckpt / f"epoch{epoch}.npy", rng.standard_normal((256, 1024), dtype=np.float32))
""",
]

# Ask before a run that moves more than this: the rides table is about twice it.
CONFIRM_READ_BYTES = 16 * 1024 * 1024
CONFIRM_READ_MESSAGE = "Large reads are faster on the lab's workers next to the bucket."


def build_lake(root: Path) -> dict[str, dict[str, str]]:
    """A ``lake`` catalog holding one table of about 34 MB, as ``STRATA_CATALOGS``.

    Random values, so Parquet cannot compress the table under the threshold.
    """
    import numpy as np
    import pyarrow as pa
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse = root / "lake-warehouse"
    warehouse.mkdir(parents=True, exist_ok=True)
    properties = {
        "type": "sql",
        "uri": f"sqlite:///{(warehouse / 'catalog.db').as_posix()}",
        "warehouse": warehouse.as_uri(),
    }
    catalog = SqlCatalog("lake", **{k: v for k, v in properties.items() if k != "type"})
    catalog.create_namespace("taxi")
    rng = np.random.default_rng(0)
    rows = 1_500_000
    data = pa.table(
        {
            "ride_id": pa.array(np.arange(rows, dtype=np.int64)),
            "fare": pa.array(rng.uniform(5, 80, rows)),
            "distance": pa.array(rng.uniform(0.5, 30, rows)),
        }
    )
    catalog.create_table("taxi.rides", data.schema).append(data)
    return {"lake": properties}


def start_worker(port: int, log_dir: Path) -> subprocess.Popen:
    """A real ``strata-worker``, so the Workers panel reports a health check it ran."""
    log = (log_dir / "worker.log").open("wb")
    proc = subprocess.Popen(
        [sys.executable, "-m", "strata.notebook.remote_executor", "--port", str(port)],
        cwd=REPO_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    _wait_for_port(proc, port, log_dir / "worker.log", "strata-worker")
    return proc


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

    Ids are backend-generated, and matching by position would silently pick
    the wrong cell if the quickstart gained a step.
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


def build_fixtures(root: Path, worker_url: str) -> tuple[Path, Path, Path]:
    """Scaffold the three fixture notebooks the web shots photograph.

    The quickstart notebook is run here. The registry and lake notebooks are
    not: their cells use the ambient ``strata`` client, so the browser script
    runs them against the live server.
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

    lake = _scaffold(root, "lake", ("pandas", "numpy"), LAKE_CELLS)
    _strata("worker", "add", str(lake), "local-dev", "--url", worker_url)
    return iris, registry, lake


def _wait_for_port(proc: subprocess.Popen, port: int, log: Path, what: str) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"{what} exited before it came up:\n{log.read_text(errors='replace')}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.25)
    proc.terminate()
    raise SystemExit(
        f"{what} did not listen on {port} within 60s:\n{log.read_text(errors='replace')}"
    )


def serve(
    storage_dir: Path, port: int, extra_env: dict[str, str] | None = None
) -> subprocess.Popen:
    env = {
        **os.environ,
        **(extra_env or {}),
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
    _wait_for_port(proc, port, log, "strata server")
    return proc


def capture_web(work_dir: Path) -> None:
    check_bundle_is_current()
    worker_port = _free_port()
    iris, registry, lake = build_fixtures(work_dir, f"http://127.0.0.1:{worker_port}/v1/execute")
    token = publish_figure(iris, work_dir)
    catalogs = build_lake(work_dir)
    worker = start_worker(worker_port, work_dir)
    port = _free_port()
    proc = serve(
        work_dir,
        port,
        {
            "STRATA_CATALOGS": json.dumps(catalogs),
            "STRATA_NOTEBOOK_CONFIRM_READ_BYTES": str(CONFIRM_READ_BYTES),
            "STRATA_NOTEBOOK_CONFIRM_READ_MESSAGE": CONFIRM_READ_MESSAGE,
        },
    )
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
                "--lake-path",
                str(lake),
                "--publication-token",
                token,
                "--out",
                str(ASSETS),
            ],
            cwd=REPO_ROOT / "frontend",
            check=True,
        )
    finally:
        for child in (proc, worker):
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Never let cleanup mask the capture error or leave the port held.
                child.kill()
                child.wait(timeout=10)


# --------------------------------------------------------------------------
# Example thumbnails (WebP, one pair per examples/ notebook)
# --------------------------------------------------------------------------

EXAMPLES_DIR = REPO_ROOT / "examples"
EXAMPLE_THUMBS = ASSETS / "examples"
# Gallery cards show them about 320 CSS px wide; twice that stays sharp on 2x screens.
THUMB_WIDTH = 640


@dataclass(frozen=True)
class ExampleShot:
    """How to photograph one example.

    ``focus`` is text in the cell whose output to take; without it the browser
    takes the first plot, else table, else markdown. ``needs`` names what a run requires
    beyond the notebook's own venv (keys of ``NEEDS``). ``in_session`` runs the
    cells in the server's session rather than with ``strata run``, for a notebook
    whose results a reopened session does not keep (a write SQL cell's).
    """

    focus: str | None = None
    needs: tuple[str, ...] = ()
    in_session: bool = False


def _llm_env() -> dict[str, str]:
    """A provider key from this shell, as the server's ``STRATA_AI_*`` settings.

    A prompt cell reads its key from the notebook's env or the server's
    settings; the shell's ``OPENAI_API_KEY`` alone reaches neither.
    """
    from strata.notebook.llm.config import _PROVIDER_DEFAULTS

    for env_var, (base_url, model) in _PROVIDER_DEFAULTS.items():
        if os.environ.get(env_var):
            return {
                "STRATA_AI_API_KEY": os.environ[env_var],
                "STRATA_AI_BASE_URL": base_url,
                "STRATA_AI_MODEL": model,
            }
    return {}


NEEDS = {
    "r": lambda: shutil.which("Rscript") is not None,
    "llm": lambda: bool(_llm_env()),
    "docker": lambda: shutil.which("docker") is not None,
    "alpaca": lambda: bool(os.environ.get("ALPACA_API_KEY")),
}

# Every notebook under examples/ needs an entry, so a new one cannot ship
# without a card image by accident.
EXAMPLE_SHOTS = {
    "agent_demo": ExampleShot(focus="The expensive step"),
    "arxiv_classifier": ExampleShot(needs=("llm",)),
    "caching": ExampleShot(),
    "data_viewer": ExampleShot(),
    "each_pipeline": ExampleShot(needs=("docker",)),
    "iris_classification": ExampleShot(),
    "keeling_curve": ExampleShot(),
    "library_cells": ExampleShot(focus="Pure module cell"),
    "loop_hill_climb": ExampleShot(),
    "markdown_showcase": ExampleShot(focus="Templated report from a DataFrame"),
    "model_variants": ExampleShot(),
    "model_variants_sweep": ExampleShot(),
    "news_alpha_trader": ExampleShot(needs=("llm", "alpaca")),
    "pandas_basics": ExampleShot(),
    "r_lm_vs_sklearn": ExampleShot(needs=("r",)),
    "r_mtcars_analysis": ExampleShot(needs=("r",)),
    "review_triage": ExampleShot(needs=("llm",)),
    "s3_mount": ExampleShot(),
    "sql_orders_report": ExampleShot(focus="Combine the two SQL results", in_session=True),
    "titanic_ml": ExampleShot(),
    "widget_playground": ExampleShot(),
}


def _env_file(path: Path) -> dict[str, str]:
    """The ``export KEY=VALUE`` lines of a file a README has you ``source``."""
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("export "):
            key, _, value = line.removeprefix("export ").partition("=")
            env[key] = shlex.split(value)[0]
    return env


def _example_services(name: str, notebook: Path, stack: contextlib.ExitStack) -> dict[str, str]:
    """Start what an example's README has you start, returning the env it exports.

    The arxiv workers stop after the run (the shot needs no worker, and its
    df-cluster port is MinIO's); each_pipeline's catalog stays up on ``stack``
    because the server checks the table when it opens the notebook.
    """
    if name == "each_pipeline":
        compose = ["docker", "compose", "-f", str(notebook / "docker-compose.yml")]
        subprocess.run([*compose, "up", "-d", "--wait", "catalog"], check=True)
        stack.callback(subprocess.run, [*compose, "down", "-v"], check=True)
        return _env_file(notebook / "lake.env")
    if name == "arxiv_classifier":
        log = notebook / "workers.log"
        with log.open("wb") as handle:
            # The ports notebook.toml names, started the way the README says.
            workers = subprocess.Popen(
                [str(notebook / "run_local_workers.sh")],
                cwd=REPO_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        def stop() -> None:
            os.killpg(workers.pid, signal.SIGINT)
            workers.wait(timeout=30)

        stack.callback(stop)
        for port in (9000, 9001):
            _wait_for_port(workers, port, log, f"worker on {port}")
    return {}


def _copy_tracked(src: Path, dest: Path) -> None:
    """Copy what git tracks under *src*: what a fresh clone has, without local
    state such as ``.strata/``, a venv or an ``renv`` library."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", str(src)],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout.split(b"\0")
    for raw in filter(None, listed):
        file = REPO_ROOT / raw.decode()
        target = dest / file.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)


# A card is the example's result on the notebook's own background, above a band of
# layered ground (strata) in one accent per example, so a row of cards reads as a set.
CARD_BACKGROUND = {"light": (239, 241, 245), "dark": (17, 17, 27)}
CARD_ACCENTS = [
    (30, 102, 245),
    (232, 89, 12),
    (112, 72, 232),
    (12, 166, 120),
    (240, 140, 0),
    (16, 152, 173),
]
THUMB_HEIGHT = THUMB_WIDTH * 10 // 16


def _strata_band(width: int, height: int, top: int, accent, background):
    """Four wavy layers from *top* down, each a deeper tint of *accent*."""
    import math

    from PIL import Image, ImageDraw

    band = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(band)
    depth = height - top
    for layer in range(4):
        tint = 0.10 + 0.09 * layer
        color = tuple(round(b + (a - b) * tint) for a, b in zip(accent, background, strict=True))
        base = top + depth * layer / 4
        amp = depth * (0.10 - 0.02 * layer)
        freq = 1.3 + 0.3 * layer
        edge = [
            (x, base + amp * math.sin(x / width * 2 * math.pi * freq + layer * 1.7))
            for x in range(0, width + 8, 8)
        ]
        draw.polygon([(0, height), *edge, (width, height)], fill=color)
    return band


def _to_thumb(png: Path, dest: Path, name: str, theme: str) -> None:
    """Set the captured result in its card: centred above the band, never cropped."""
    import zlib

    from PIL import Image

    scale = 2
    width, height = THUMB_WIDTH * scale, THUMB_HEIGHT * scale
    background = CARD_BACKGROUND[theme]
    accent = CARD_ACCENTS[zlib.crc32(name.encode()) % len(CARD_ACCENTS)]
    card = _strata_band(width, height, round(height * 0.80), accent, background)
    pad = round(width * 0.04)
    box = (width - 2 * pad, round(height * 0.78) - pad)
    with Image.open(png) as image:
        result = image.convert("RGBA")
        result.thumbnail(box, Image.Resampling.LANCZOS)
        left = (width - result.width) // 2
        top = pad + (box[1] - result.height) // 2
        card.paste(result, (left, top), result)
    card.resize((THUMB_WIDTH, THUMB_HEIGHT), Image.Resampling.LANCZOS).save(
        dest, "WEBP", quality=86, method=6
    )


def capture_examples(work_dir: Path, names: list[str] | None) -> list[Path]:
    """Run each example in a scratch copy, then photograph it in both themes.

    An example whose requirements are missing, or whose run fails, keeps its
    previous thumbnail: a card should never show an error.
    """
    check_bundle_is_current()
    available = sorted(p.parent.name for p in EXAMPLES_DIR.glob("*/notebook.toml"))
    unlisted = sorted(set(available) - EXAMPLE_SHOTS.keys())
    if unlisted:
        raise SystemExit(f"add an EXAMPLE_SHOTS entry for: {', '.join(unlisted)}")
    unknown = sorted(set(names or ()) - set(available))
    if unknown:
        raise SystemExit(f"no example named: {', '.join(unknown)}")

    root = work_dir / "examples"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    raw = work_dir / "example-pngs"
    shutil.rmtree(raw, ignore_errors=True)
    raw.mkdir(parents=True)
    manifest, skipped = [], {}
    written: list[Path] = []
    server_env: dict[str, str] = {}
    with contextlib.ExitStack() as services:
        for name in names or available:
            shot = EXAMPLE_SHOTS[name]
            missing = [need for need in shot.needs if not NEEDS[need]()]
            if missing:
                skipped[name] = f"needs {', '.join(missing)}"
                continue
            notebook = root / name
            _copy_tracked(EXAMPLES_DIR / name, notebook)
            with contextlib.ExitStack() as run_only:
                try:
                    env = _example_services(name, notebook, run_only)
                except (subprocess.CalledProcessError, SystemExit) as exc:
                    skipped[name] = f"its services did not start: {exc}"
                    continue
                if "llm" in shot.needs or shot.in_session:
                    # A headless `strata run` reads no server settings, so an LLM key
                    # would have to sit in the notebook's env on disk. The server
                    # takes it from its environment: run these cells there.
                    server_env.update(env)
                    if "llm" in shot.needs:
                        server_env.update(_llm_env())
                    manifest.append(
                        {"name": name, "path": str(notebook), "focus": shot.focus, "run": True}
                    )
                    continue
                if name == "each_pipeline":
                    # Its catalog outlives the run: hand the teardown to the outer stack.
                    services.push(run_only.pop_all())
                run = subprocess.run(
                    [sys.executable, "-m", "strata.cli", "run", str(notebook)],
                    cwd=REPO_ROOT,
                    env={**os.environ, **env},
                )
            server_env.update(env)
            if run.returncode != 0:
                skipped[name] = f"`strata run` exited {run.returncode}"
                continue
            manifest.append({"name": name, "path": str(notebook), "focus": shot.focus})

        manifest_path = work_dir / "examples.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        if manifest:
            port = _free_port()
            proc = serve(root, port, server_env)
            try:
                subprocess.run(
                    [
                        "node",
                        "scripts/capture-example-thumbs.mjs",
                        "--base-url",
                        f"http://127.0.0.1:{port}",
                        "--manifest",
                        str(manifest_path),
                        "--out",
                        str(raw),
                    ],
                    cwd=REPO_ROOT / "frontend",
                    check=True,
                )
            finally:
                proc.terminate()
                proc.wait(timeout=30)
    EXAMPLE_THUMBS.mkdir(parents=True, exist_ok=True)
    for png in sorted(raw.glob("*.png")):
        dest = EXAMPLE_THUMBS / f"{png.stem}.webp"
        name, _, theme = png.stem.rpartition("-")
        _to_thumb(png, dest, name, theme)
        written.append(dest)

    for name, reason in skipped.items():
        print(f"skipped {name}: {reason}", file=sys.stderr)
    return written


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("all", "tui", "web", "anim", "examples"), default="all")
    parser.add_argument(
        "--example",
        action="append",
        dest="examples",
        help="With --only examples: photograph just this one (repeatable)",
    )
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
    if args.only in ("all", "examples"):
        written += capture_examples(Path(args.work_dir), args.examples)

    for path in written:
        print(f"{path.relative_to(REPO_ROOT)}  {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
