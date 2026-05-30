"""Cross-language R cell integration tests — the #59 capstone suite.

This file pins the end-to-end "Python ↔ R" story: Python cells produce
artifacts that R cells consume via Arrow IPC, R cells produce artifacts
that downstream Python cells read back. All gated on Rscript + the R
``arrow`` package being available; tests skip cleanly otherwise.

First slice (this PR): one smoke test that exercises the full
Py → R → Py loop. Cross-language error shapes, provenance / cache
behaviour, annotations, and real-renv restore land in follow-up PRs.
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
# Note: deliberately NOT ``@pytest.mark.integration``. That marker
# opts out of conftest's autouse ``fast_notebook_env`` override, which
# replaces the Python cell's ``uv run python harness.py`` invocation
# with a direct call to the dev interpreter (so the test gets pandas /
# pyarrow without a per-notebook ``uv sync``). The R harness path
# (``_run_r_harness``) is *not* overridden — it always shells out to
# real ``Rscript`` — so leaving the autouse override active gives us
# the natural mix: Python cells run against the dev venv, R cells run
# against the system R install. The renv-managed library tier that
# the issue describes lands in a follow-up PR.


@pytest.mark.asyncio
async def test_py_to_r_to_py_arrow_roundtrip(r_notebook):
    """A Python DataFrame round-trips through an R cell back into Python.

    Three cells:
      c1 (Python) — build a ``pandas.DataFrame`` and bind it as ``df``.
      c2 (R)      — read ``df`` as a ``data.frame``, append a derived
                    column, bind the result as ``df_r``.
      c3 (Python) — read ``df_r`` back (as a pandas DataFrame via the
                    Arrow IPC handoff) and compute a scalar.

    What this pins:
      * R harness ingests an Arrow IPC artifact produced by the Python
        harness (``arrow::read_ipc_stream`` round-trips
        column names + types).
      * R harness emits an Arrow IPC artifact via
        ``arrow::write_ipc_stream``, picked up by the Python harness on
        the next cell with no extra coercion.
      * DAG analysis on a mixed-language notebook wires Python ↔ R
        upstream/downstream edges off bare variable names — no
        language-specific annotation needed.

    Materialisation: each cell runs with ``materialize_upstreams=True``
    (the default), so calling ``execute_cell("c3", ...)`` would also
    walk back through c2 and c1. Spelling each ``execute_cell`` out
    here keeps the failure-localisation obvious — if c2 breaks, the
    test surfaces it on the c2 assertion rather than a confusing
    cascade error on c3.
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
    """An R cell that produces a non-tabular value stores it as RDS;
    a downstream Python cell consuming it fails with the structured
    ``StrataRArtifactError`` instead of a ``NameError``.

    Two cells:
      c1 (R)      — build a classed list (``structure(list(...),
                    class = ...)``). Not a ``data.frame``, not a bare
                    list (``is.object()`` is TRUE), so harness.R's
                    serializer falls through to the ``write_rds``
                    tier with ``content_type =
                    "application/x-r-rds"``.
      c2 (Python) — references ``model``. The harness's
                    ``deserialize_inputs`` hits the registered
                    RDS handler, which raises ``StrataRArtifactError``
                    with the upstream variable name. The cell fails
                    *before* the body runs — no chance for a confusing
                    ``NameError: 'model'`` to surface.

    This is the cross-language counterpart to the unit tests in
    ``test_serializer.py::TestRdsArtifactRefusal`` and
    ``test_harness.py::TestHarnessRdsInput``. Those pin the
    deserializer + harness re-raise in isolation; this test pins the
    full notebook-level flow that #58 was designed for.
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
    # Critical regression assertion — pre-#58 (and pre-#72 fix-up),
    # the deserialize error was swallowed and the cell body raised
    # ``NameError: 'model'`` instead.
    assert "NameError" not in err, f"regressed to NameError: {err!r}"


@pytest.mark.asyncio
async def test_r_cell_mount_injects_path_and_reads_file(r_notebook, tmp_path):
    """``# @mount data file://<path>`` binds ``data`` inside an R cell
    the same way it binds inside a Python cell.

    The R harness's ``inject_mounts`` (in ``harness.R``) assigns each
    mount-name → ``local_path`` string into the cell environment. R
    has no native ``Path`` type, so the binding is a plain character
    vector — ``file.path(data, "x.txt")`` constructs the full path
    just like ``data / "x.txt"`` would in Python.

    This test exercises:
      * Annotation parsing on R cells (the same parser handles both
        languages — see #54's dispatch refactor).
      * Mount resolution + injection through ``_resolve_mounts`` /
        ``inject_mounts``.
      * ``readLines`` against a file under the mount root.

    Read-only ``file://`` mount — no need to exercise rw / cloud
    scheme variants here; those have dedicated coverage in
    ``test_mounts.py`` and the e2e_mounts_* files.
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


# ---------------------------------------------------------------------------
# Error-shape tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_syntax_error_surfaces_as_failure(r_notebook):
    """An unparseable R cell fails the cell, not the harness.

    Distinct from a runtime ``stop()`` — R signals parse errors via
    ``parse()`` before any user code runs. harness.R's outer
    ``tryCatch`` around ``parse(text = source_text)`` catches it,
    records the message + ``sys.calls()`` traceback, and writes
    ``success: false`` to ``harness-result.json``.

    Runtime errors are pinned separately in
    ``test_language_r_executor.py::TestExecuteSimpleRCell::test_runtime_error_surfaces_as_failure``.
    """
    src = "x <-"  # incomplete expression, no RHS
    _, session = r_notebook(cells=[("c1", None, src, "r")])
    executor = CellExecutor(session)

    result = await executor.execute_cell("c1", src)

    assert result.success is False
    # The exact R parse-error wording varies across R versions but
    # always mentions ``unexpected`` or ``end of input``. Either is
    # a sufficient signal that the failure surfaced from R's parser
    # rather than from some harness layer.
    err = (result.error or "").lower()
    assert "unexpected" in err or "end of" in err, (
        f"expected an R parse-error message, got: {result.error!r}"
    )


# ---------------------------------------------------------------------------
# Provenance / cache behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_cell_cache_hits_on_unchanged_re_run(r_notebook):
    """Running the same R cell twice: second run is a cache hit.

    Provenance is language-agnostic (``(source_hash, env_hash,
    sorted_inputs)``) so R cells use the exact same cache path as
    Python cells. This test is the R analogue of
    ``test_executor.py``'s display-cache-hit cases — pins that the
    R execution flow ends with a stored artifact + cached metadata,
    and that a second invocation skips the harness spawn.

    Why a downstream consumer cell: the executor's cache lookup
    keys off ``derive_subkey(provenance_hash, first_consumed_var)``
    when ``consumed_variables`` is non-empty, matching the per-var
    artifact-store path used at write time. A leaf cell with no
    downstream consumer falls through the alternate ``find_cached
    (provenance_hash)`` branch where storage and lookup keys
    differ, and the second run looks like a cache miss for
    bookkeeping reasons rather than a real one. The smoke + RDS
    + mount cases earlier in this file all run as the *producer*
    side of a multi-cell chain, so this is the only spot the
    distinction matters.
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
    # harness.R's JSON tier formats atomic scalars via ``format()`` —
    # ``7`` reads back as the string ``"7"``, not the int.
    assert first.outputs["value"]["preview"] == "7"

    second = await executor.execute_cell("c1", src)
    assert second.success is True, second.error
    assert second.cache_hit is True
    # ``CellExecutionResult.outputs`` is intentionally empty on the
    # cache-hit branch (executor.py L1281) — the artifact is already
    # in the store, so the result is a thin pointer. The
    # ``execution_method == "cached"`` field is the cleanest
    # signal that the harness was skipped.
    assert second.execution_method == "cached"


@pytest.mark.asyncio
async def test_r_cell_source_change_invalidates_cache(r_notebook):
    """Editing the cell source flips the provenance hash → cache miss.

    The cell ID stays the same; only the body changes. Pins that
    ``source_hash`` participates in provenance for R cells the same
    way it does for Python cells (and that the R harness re-runs on
    the new body).
    """
    # Downstream Python cell pins ``value`` as a consumed variable so
    # the cache lookup uses ``derive_subkey(provenance, "value")`` —
    # matching the per-var write path. See the unchanged-rerun test
    # above for the why.
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


# ---------------------------------------------------------------------------
# Annotation tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_cell_renv_lock_change_invalidates_cache(r_notebook):
    """Editing ``renv.lock`` in the notebook dir invalidates R cell caches.

    Pins the #59 acceptance criterion "renv.lock change invalidates
    all R cells (env hash changed)". The actual env-hash extension
    is unit-tested in ``test_env.py`` (no R needed); this end-to-end
    test confirms the executor's R cell path picks up the new hash
    and treats the next run as a cache miss.

    No real ``renv::restore()`` happens — the system R + the
    ``arrow`` / ``jsonlite`` packages installed via the ``r-tests``
    CI step are what the cell actually loads against. We're only
    pinning that the lockfile's *content* feeds provenance, not
    that the libraries it pins are the ones loaded.
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

    # Write a minimal renv.lock with one package. Bytes don't have
    # to be a real renv lockfile — env-hash computation hashes the
    # bytes, not the schema.
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
    """``# @env KEY=value`` injects into the R cell's Sys.getenv().

    The annotation parser (``annotations.py``) is language-agnostic
    — it scans the leading ``#``-comment block of *any* cell — and
    harness.R's ``Sys.setenv`` block applies the manifest's ``env``
    dict to the R process before the cell body runs. This test pins
    that contract end-to-end: an annotation declared on an R cell
    is readable via ``Sys.getenv`` from inside the same cell.
    """
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
    # PNG magic number — proves the base64 round-trips to real image bytes.
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.asyncio
async def test_r_cell_base_graphics_emitted_as_png_display(r_notebook):
    """A base-graphics plot in an R cell becomes an image/png display.

    Base graphics (``plot``) draw straight to the harness's capture
    device during ``eval`` (the ``plot.new`` hook marks the page as
    real), so the page is re-homed to ``__display__0.png`` and
    surfaced through the same ``_store_display_outputs`` chain the
    Python matplotlib path uses. No extra R packages needed — this is
    the CI-safe core of #80.
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
    """Two plots in one R cell produce two ordered image/png displays."""
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

    Guards the capture device's blank-page handling: the trailing
    blank PNG an unused device writes on close must not be mistaken
    for a real plot, and closing the device must not leak
    ``null device 1`` into the captured stdout.
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


@skip_if_no_r_ggplot2
@pytest.mark.asyncio
async def test_r_cell_ggplot_emitted_as_png_display(r_notebook):
    """A bare trailing ggplot object auto-renders to an image/png display.

    The harness auto-prints the visible value of a top-level
    expression when it's plot-like (``inherits(x, "ggplot")``), so a
    notebook-style last-line ``p`` renders without an explicit
    ``print(p)`` — mirroring the REPL. ``print.ggplot`` draws via
    ``grid.newpage`` (hooked, marks the page real). Gated on ggplot2
    being installed; skips cleanly in CI where it isn't.
    """
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
    """A real ``renv::restore`` from a committed lockfile populates the
    project library, and an R cell then runs against it — no mocks.

    This is the renv capstone the #59 suite deferred: ``test_renv_sync.py``
    monkeypatches ``subprocess.run``, and the plain ``r_notebook`` fixture
    runs R cells against the system library. Here the committed
    ``renv_jsonlite`` scaffold ships a lockfile but no built library, so
    ``_renv_sync`` has to do a genuine restore. Pinned to ``jsonlite``
    only, which restores in seconds from a binary.

    Asserts, in order:
      1. The scaffold has no project library yet (restore is the work).
      2. ``_renv_sync`` reports success.
      3. ``jsonlite`` is now installed under ``renv/library`` — proof the
         restore actually populated the project-scoped library rather than
         silently falling back to the system one.
      4. An R cell that loads jsonlite executes against that library.
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
