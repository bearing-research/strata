"""Cross-language R cell integration tests.

Python cells produce artifacts R cells read via Arrow IPC, and back. Every test
skips unless Rscript and the R ``arrow`` package are available.
"""

from __future__ import annotations

import base64

import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.writer import _renv_sync
from tests.notebook.conftest import (
    skip_if_no_r,
    skip_if_no_r_arrow,
    skip_if_no_r_ggplot2,
)

pytestmark = [skip_if_no_r, skip_if_no_r_arrow]
# Deliberately not ``@pytest.mark.integration``: that marker opts out of the
# autouse ``fast_notebook_env`` override, which runs Python cells on the dev
# interpreter. The R harness always shells out to real ``Rscript``, so Python
# cells use the dev venv and R cells the system R install.


@pytest.mark.asyncio
async def test_py_to_r_to_py_arrow_roundtrip(r_notebook):
    """Python -> R -> Python over Arrow IPC, with DAG edges wired off bare names.

    Each cell is executed explicitly so a failure points at the cell that broke,
    not at a cascade error on c3.
    """
    py_c1 = "import pandas as pd\ndf = pd.DataFrame({'x': [1, 2, 3], 'y': [10, 20, 30]})\n"
    r_c2 = "df_r <- df\ndf_r$z <- df_r$x + df_r$y\n"
    py_c3 = "total = int(df_r['z'].sum())\n"

    _, session = r_notebook(
        cells=[
            ("c1", None, py_c1, "python"),
            ("c2", "c1", r_c2, "r"),
            ("c3", "c2", py_c3, "python"),
        ]
    )
    executor = CellExecutor(session)

    r1 = await executor.execute_cell("c1", py_c1)
    assert r1.success is True, r1.error
    assert "df" in r1.outputs
    assert r1.outputs["df"]["content_type"] == "arrow/ipc"

    r2 = await executor.execute_cell("c2", r_c2)
    assert r2.success is True, r2.error
    assert "df_r" in r2.outputs
    assert r2.outputs["df_r"]["content_type"] == "arrow/ipc"
    # R round-trip preserves the three-row shape + adds the derived
    # column; ``rows`` / ``columns`` come from harness.R's write_arrow
    # metadata.
    assert r2.outputs["df_r"]["rows"] == 3
    assert r2.outputs["df_r"]["columns"] == 3

    r3 = await executor.execute_cell("c3", py_c3)
    assert r3.success is True, r3.error
    # 11 + 22 + 33 == 66.
    assert r3.outputs["total"]["preview"] == 66


@pytest.mark.asyncio
async def test_r_only_rds_artifact_rejected_by_downstream_python_cell(r_notebook):
    """An RDS-only R value fails a Python consumer with ``StrataRArtifactError``.

    c1 builds a classed list, which serializes as RDS. c2 must fail before its body
    runs, so the error is the structured one rather than ``NameError: 'model'``.
    """
    r_c1 = 'model <- structure(list(coef = 1.5, intercept = 0.0), class = "fit_model")\n'
    py_c2 = "score = model['coef']\n"

    _, session = r_notebook(
        cells=[
            ("c1", None, r_c1, "r"),
            ("c2", "c1", py_c2, "python"),
        ]
    )
    executor = CellExecutor(session)

    r1 = await executor.execute_cell("c1", r_c1)
    assert r1.success is True, r1.error
    assert r1.outputs["model"]["content_type"] == "application/x-r-rds"
    # The R harness tags r_only=true on the payload so downstream
    # consumers can decide before opening the blob.
    assert r1.outputs["model"].get("r_only") is True

    r2 = await executor.execute_cell("c2", py_c2)
    assert r2.success is False, "Python cell must reject R-only RDS upstream"
    err = r2.error or ""
    # The structured error names the variable + suggests the fix.
    assert "model" in err, f"variable name missing from error: {err!r}"
    assert "saveRDS" in err, f"saveRDS hint missing: {err!r}"
    assert "data.frame" in err, f"re-export hint missing: {err!r}"
    # The deserialize error must not be swallowed, leaving the cell body to raise
    # ``NameError: 'model'``.
    assert "NameError" not in err, f"regressed to NameError: {err!r}"


@pytest.mark.asyncio
async def test_python_only_pickle_artifact_rejected_by_downstream_r_cell(r_notebook):
    """A pickle-only Python value fails an R consumer with a structured result.

    The R harness must write a ``success: false`` envelope rather than abort
    Rscript, which would leave the executor scraping stderr.
    """
    py_c1 = "pyobj = {1, 2, 3}\n"
    r_c2 = "n <- length(pyobj)\n"

    _, session = r_notebook(
        cells=[
            ("c1", None, py_c1, "python"),
            ("c2", "c1", r_c2, "r"),
        ]
    )
    executor = CellExecutor(session)

    r1 = await executor.execute_cell("c1", py_c1)
    assert r1.success is True, r1.error
    assert r1.outputs["pyobj"]["content_type"] == "pickle/object"

    r2 = await executor.execute_cell("c2", r_c2)
    assert r2.success is False, "R cell must reject Python pickle upstream"
    err = r2.error or ""
    # The structured error names the variable + suggests the fix.
    assert "pyobj" in err, f"variable name missing from error: {err!r}"
    assert "pickle" in err, f"content-type hint missing: {err!r}"
    assert "DataFrame" in err or "Arrow" in err, f"re-export hint missing: {err!r}"
    # The error must come from the structured envelope, not the missing-manifest
    # fallback an aborted Rscript hits.
    assert "without producing a result manifest" not in err, (
        f"regressed to missing-manifest fallback: {err!r}"
    )
    # R's own "not found" must not leak: the read fails cleanly before the cell
    # body references the variable.
    assert "not found" not in err, f"regressed to R NameError: {err!r}"


@pytest.mark.asyncio
async def test_python_numpy_array_into_r_warns_on_shape_flattening(r_notebook):
    """A 2-D ndarray reaches R as a flat column, and the harness warns in stderr.

    R's Arrow reader cannot rebuild the tensor shape, so the cell runs but loses
    fidelity; the warning is the only signal.
    """
    py_c1 = "import numpy as np\narr = np.array([[1, 2], [3, 4]])\n"
    r_c2 = "n <- nrow(arr)\n"

    _, session = r_notebook(
        cells=[
            ("c1", None, py_c1, "python"),
            ("c2", "c1", r_c2, "r"),
        ]
    )
    executor = CellExecutor(session)

    r1 = await executor.execute_cell("c1", py_c1)
    assert r1.success is True, r1.error
    assert r1.outputs["arr"]["content_type"] == "arrow/ipc"

    r2 = await executor.execute_cell("c2", r_c2)
    # The cell still succeeds: flattening is a fidelity change, not an error.
    assert r2.success is True, r2.error
    warn = r2.stderr or ""
    assert "arr" in warn, f"variable name missing from warning: {warn!r}"
    assert "tensor" in warn, f"shape missing from warning: {warn!r}"
    assert "flattened" in warn, f"flatten hint missing from warning: {warn!r}"


@pytest.mark.asyncio
async def test_r_cell_mount_injects_path_and_reads_file(r_notebook, tmp_path):
    """``# @mount`` binds the mount name to a path string inside an R cell.

    R has no ``Path`` type, so the binding is a character vector usable with
    ``file.path``.
    """
    mount_dir = tmp_path / "shared_data"
    mount_dir.mkdir()
    (mount_dir / "greeting.txt").write_text("hello from a mount\n", encoding="utf-8")

    r_src = (
        f'# @mount data file://{mount_dir}\ncontent <- readLines(file.path(data, "greeting.txt"))\n'
    )

    _, session = r_notebook(cells=[("c1", None, r_src, "r")])
    executor = CellExecutor(session)

    r1 = await executor.execute_cell("c1", r_src)
    assert r1.success is True, r1.error
    # ``readLines`` returns a character vector; harness.R's JSON tier
    # writes it as a 1-element array (json/object) or scalar depending
    # on auto_unbox. The preview faithfully reproduces the value.
    assert "hello from a mount" in str(r1.outputs["content"]["preview"])


# Error-shape tests


@pytest.mark.asyncio
async def test_r_syntax_error_surfaces_as_failure(r_notebook):
    """An unparseable R cell fails the cell, not the harness.

    Parse errors happen before any user code runs, unlike a runtime ``stop()``.
    """
    src = "x <-"  # incomplete expression, no RHS
    _, session = r_notebook(cells=[("c1", None, src, "r")])
    executor = CellExecutor(session)

    result = await executor.execute_cell("c1", src)

    assert result.success is False
    # R's parse-error wording varies by version but always mentions
    # ``unexpected`` or ``end of input``; either shows the failure came from R's
    # parser rather than a harness layer.
    err = (result.error or "").lower()
    assert "unexpected" in err or "end of" in err, (
        f"expected an R parse-error message, got: {result.error!r}"
    )


# Provenance / cache behaviour


@pytest.mark.asyncio
async def test_r_cell_cache_hits_on_unchanged_re_run(r_notebook):
    """Running the same R cell twice: the second run is a cache hit.

    The downstream consumer matters: a leaf cell is stored and looked up under
    different keys, so its second run looks like a miss for bookkeeping reasons.
    """
    src = "value <- 7"
    py_downstream = "scaled = value\n"
    _, session = r_notebook(
        cells=[
            ("c1", None, src, "r"),
            ("c2", "c1", py_downstream, "python"),
        ]
    )
    executor = CellExecutor(session)

    first = await executor.execute_cell("c1", src)
    assert first.success is True, first.error
    assert first.cache_hit is False
    # harness.R's JSON tier formats atomic scalars via ``format()``: ``7`` reads
    # back as the string ``"7"``, not the int.
    assert first.outputs["value"]["preview"] == "7"

    second = await executor.execute_cell("c1", src)
    assert second.success is True, second.error
    assert second.cache_hit is True
    # ``outputs`` is empty on the cache-hit branch (the artifact is already in
    # the store), so ``execution_method == "cached"`` is the signal that the
    # harness was skipped.
    assert second.execution_method == "cached"


@pytest.mark.asyncio
async def test_r_cell_source_change_invalidates_cache(r_notebook):
    """Editing the source of the same cell id gives a cache miss."""
    # The downstream Python cell makes ``value`` a consumed variable, so the cache
    # lookup uses ``derive_subkey(provenance, "value")`` like the per-var write path.
    src_v1 = "value <- 1"
    src_v2 = "value <- 2"
    py_downstream = "doubled = value\n"
    _, session = r_notebook(
        cells=[
            ("c1", None, src_v1, "r"),
            ("c2", "c1", py_downstream, "python"),
        ]
    )
    executor = CellExecutor(session)

    first = await executor.execute_cell("c1", src_v1)
    assert first.success is True, first.error
    assert first.cache_hit is False
    # harness.R's JSON tier formats atomic scalars as strings.
    assert first.outputs["value"]["preview"] == "1"

    second = await executor.execute_cell("c1", src_v2)
    assert second.success is True, second.error
    assert second.cache_hit is False, "source change must invalidate cache"
    assert second.outputs["value"]["preview"] == "2"


# Annotation tests


@pytest.mark.asyncio
async def test_r_cell_renv_lock_change_invalidates_cache(r_notebook):
    """Editing ``renv.lock`` changes the env hash, so the next R run is a cache miss.

    No ``renv::restore()`` runs: only the lockfile's content feeding provenance is
    pinned, not which libraries load.
    """
    src = "value <- 99"
    py_downstream = "passthrough = value\n"
    notebook_dir, session = r_notebook(
        cells=[
            ("c1", None, src, "r"),
            ("c2", "c1", py_downstream, "python"),
        ]
    )
    executor = CellExecutor(session)

    # A minimal renv.lock: the env hash covers the bytes, not the schema.
    renv_lock = notebook_dir / "renv.lock"
    renv_lock.write_text('{"R": {"Version": "4.4.0"}, "Packages": {"arrow": "1.0"}}\n')

    first = await executor.execute_cell("c1", src)
    assert first.success is True, first.error
    assert first.cache_hit is False

    # Same source + same renv.lock → cache hit.
    second = await executor.execute_cell("c1", src)
    assert second.cache_hit is True, "no-change re-run must hit the cache"

    # Edit renv.lock (different pinned arrow version) → env_hash
    # changes → provenance hash changes → cache miss.
    renv_lock.write_text('{"R": {"Version": "4.4.0"}, "Packages": {"arrow": "2.0"}}\n')
    third = await executor.execute_cell("c1", src)
    assert third.cache_hit is False, "renv.lock edit must invalidate cache"


@pytest.mark.asyncio
async def test_r_cell_env_annotation_visible_to_rscript(r_notebook):
    """``# @env KEY=value`` on an R cell is visible to ``Sys.getenv`` in that cell."""
    src = "# @env STRATA_TEST_VAR=hello-from-annotation\nvalue <- Sys.getenv('STRATA_TEST_VAR')\n"
    _, session = r_notebook(cells=[("c1", None, src, "r")])
    executor = CellExecutor(session)

    result = await executor.execute_cell("c1", src)

    assert result.success is True, result.error
    assert result.outputs["value"]["preview"] == "hello-from-annotation"


def _assert_png_display(display: dict) -> None:
    """A display payload is a persisted image/png with valid PNG bytes."""
    assert display["content_type"] == "image/png"
    assert display["bytes"] > 0
    assert display["width"] == 800
    assert display["height"] == 600
    assert display.get("artifact_uri"), "display should be persisted as an artifact"
    data_url = display["inline_data_url"]
    assert data_url.startswith("data:image/png;base64,")
    raw = base64.b64decode(data_url.split(",", 1)[1])
    # PNG magic number: the base64 round-trips to real image bytes.
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.asyncio
async def test_r_cell_base_graphics_emitted_as_png_display(r_notebook):
    """A base-graphics plot in an R cell becomes an image/png display.

    Needs no extra R packages, so this is the CI-safe plot case.
    """
    src = "plot(1:10, (1:10)^2, main = 'quadratic')\n"
    _, session = r_notebook(cells=[("c1", None, src, "r")])
    executor = CellExecutor(session)

    result = await executor.execute_cell("c1", src)

    assert result.success is True, result.error
    assert len(result.display_outputs) == 1
    _assert_png_display(result.display_outputs[0])


@pytest.mark.asyncio
async def test_r_cell_multiple_plots_emit_ordered_displays(r_notebook):
    src = "plot(1:10)\nhist(c(1, 1, 2, 3, 3, 3))\n"
    _, session = r_notebook(cells=[("c1", None, src, "r")])
    executor = CellExecutor(session)

    result = await executor.execute_cell("c1", src)

    assert result.success is True, result.error
    assert len(result.display_outputs) == 2
    for display in result.display_outputs:
        _assert_png_display(display)
    # Distinct artifacts, persisted in draw order.
    assert result.display_outputs[0]["artifact_uri"] != result.display_outputs[1]["artifact_uri"]


@pytest.mark.asyncio
async def test_r_cell_without_plot_emits_no_display(r_notebook):
    """A non-plotting R cell emits no displays and keeps stdout clean.

    The blank PNG an unused device writes on close must not count as a plot, and
    closing it must not leak ``null device 1`` into stdout.
    """
    src = "x <- mean(1:100)\ncat('no plots here\\n')\n"
    _, session = r_notebook(cells=[("c1", None, src, "r")])
    executor = CellExecutor(session)

    result = await executor.execute_cell("c1", src)

    assert result.success is True, result.error
    assert result.display_outputs == []
    assert result.stdout.strip() == "no plots here"
    # R scalar previews come back as formatted strings (harness.R write_json).
    assert result.outputs["x"]["preview"] == "50.5"


@pytest.mark.asyncio
async def test_r_cell_trailing_expression_auto_prints(r_notebook):
    """A bare trailing expression auto-prints, not only plot-like values."""
    src = "df <- data.frame(a = c(1L, 2L, 3L))\nsum(df$a)\n"
    _, session = r_notebook(cells=[("c1", None, src, "r")])
    executor = CellExecutor(session)

    result = await executor.execute_cell("c1", src)

    assert result.success is True, result.error
    assert result.display_outputs == []
    # sum(1:3) == 6: `[1] 6` is auto-printed; the assignment stays invisible.
    assert "6" in result.stdout


@pytest.mark.asyncio
async def test_r_cell_grid_draw_captured_as_png_display(r_notebook):
    """``grid.draw()`` without ``grid.newpage`` is still captured as a PNG.

    Capture keys off the files the device wrote, not a page count that
    ``grid.draw()`` never bumps.
    """
    src = "library(grid)\ngrid.draw(circleGrob(r = 0.3))\n"
    _, session = r_notebook(cells=[("c1", None, src, "r")])
    executor = CellExecutor(session)

    result = await executor.execute_cell("c1", src)

    assert result.success is True, result.error
    assert len(result.display_outputs) == 1
    _assert_png_display(result.display_outputs[0])


@skip_if_no_r_ggplot2
@pytest.mark.asyncio
async def test_r_cell_ggplot_emitted_as_png_display(r_notebook):
    """A bare trailing ggplot object renders without ``print(p)``; skips without ggplot2."""
    src = (
        "library(ggplot2)\n"
        "df <- data.frame(x = 1:10, y = (1:10)^2)\n"
        "p <- ggplot(df, aes(x, y)) + geom_point()\n"
        "p\n"
    )
    _, session = r_notebook(cells=[("c1", None, src, "r")])
    executor = CellExecutor(session)

    result = await executor.execute_cell("c1", src)

    assert result.success is True, result.error
    assert len(result.display_outputs) == 1
    _assert_png_display(result.display_outputs[0])


@pytest.mark.asyncio
async def test_renv_restore_populates_project_library_and_runs_cell(r_notebook_renv):
    """A real ``renv::restore`` populates the project library and a cell runs on it.

    No mocks: the scaffold ships a lockfile but no library, so the restore does the
    work. Checking ``renv/library`` proves it did not fall back to the system library.
    """
    src = "library(jsonlite)\nout <- as.character(toJSON(list(ok = TRUE), auto_unbox = TRUE))\n"
    notebook_dir, session = r_notebook_renv(cells=[("c1", None, src, "r")])

    lib_root = notebook_dir / "renv" / "library"
    assert not (lib_root.exists() and list(lib_root.rglob("jsonlite"))), (
        "fixture should ship no pre-built library — restore is what's under test"
    )

    assert _renv_sync(notebook_dir) is True, "renv::restore() should succeed"

    assert list(lib_root.rglob("jsonlite")), (
        "renv::restore must install jsonlite into the project library"
    )

    executor = CellExecutor(session)
    result = await executor.execute_cell("c1", src)

    assert result.success is True, result.error
    assert result.outputs["out"]["preview"] == '{"ok":true}'
