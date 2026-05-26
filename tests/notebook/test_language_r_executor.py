"""Tests for the R cell executor (``_execute_r_cell`` + ``_RExecutor``).

Two tiers, matching ``test_language_r_analyzer.py``:

- **Unit tests** monkeypatch ``shutil.which`` / ``_run_r_harness`` to
  cover the wrapper, the registry wiring, and the dispatch surface
  without needing R installed.
- **Integration tests** spawn real ``Rscript`` and assert end-to-end
  behaviour against the unmodified ``harness.R``. Gated on Rscript
  availability so they skip cleanly on dev machines / CI variants
  without R.

The capstone real-renv + cross-language Arrow handoff tests land
with #59.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.languages import get_language_executor
from strata.notebook.languages.r.executor import _RExecutor
from strata.notebook.models import CellLanguage
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from tests.notebook.conftest import skip_if_no_r as rscript_available


def _make_r_notebook(tmp_path: Path, *, cells: list[tuple[str, str | None, str]]):
    """Build an R notebook with the given (cell_id, after_id, source) cells.

    Returns ``(notebook_dir, session)``. Every cell's ``language`` is forced
    to ``CellLanguage.R`` so dispatch routes through the R executor.
    """
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    notebook_dir = create_notebook(tmp_path, "R Executor Test", initialize_environment=False)
    for cell_id, after_id, source in cells:
        add_cell_to_notebook(notebook_dir, cell_id, after_id, language="r")
        write_cell(notebook_dir, cell_id, source)

    notebook_state = parse_notebook(notebook_dir)
    for cell in notebook_state.cells:
        cell.language = CellLanguage.R
    session = NotebookSession(notebook_state, notebook_dir)
    return notebook_dir, session


# ---------------------------------------------------------------------------
# Registry + dispatch
# ---------------------------------------------------------------------------


class TestRegistryWiring:
    """``_RExecutor`` registers itself at import time."""

    def test_r_executor_is_registered(self):
        adapter = get_language_executor(CellLanguage.R)
        assert isinstance(adapter, _RExecutor)

    def test_behaviour_flags(self):
        """R goes through the generic provenance + cache pipeline."""
        adapter = get_language_executor(CellLanguage.R)
        assert adapter.skips_execution_provenance is False
        assert adapter.has_alternate_cache_scheme is False

    def test_is_batchable_returns_false(self):
        """Phase 1 (#57) — R is single-shot, batching deferred."""
        adapter = get_language_executor(CellLanguage.R)
        sentinel = object()
        assert adapter.is_batchable(sentinel, sentinel) is False


class TestExecutorPaths:
    """``r_harness_path`` is set up at executor construction."""

    def test_executor_has_r_harness_path(self, tmp_path):
        _, session = _make_r_notebook(tmp_path, cells=[("c1", None, "x <- 1")])
        executor = CellExecutor(session)
        assert executor.r_harness_path.name == "harness.R"
        assert executor.r_harness_path.exists()

    def test_dispatch_routes_through_r_executor(self, monkeypatch, tmp_path):
        """``_materialize_cell`` looks up the language executor and forwards."""
        _, session = _make_r_notebook(tmp_path, cells=[("c1", None, "x <- 1")])
        executor = CellExecutor(session)

        calls: list[str] = []

        async def fake_execute_r(self, cell_id, source, *args, **kwargs):
            calls.append(cell_id)
            from strata.notebook.executor import CellExecutionResult

            return CellExecutionResult(cell_id=cell_id, success=True, duration_ms=0.0)

        monkeypatch.setattr(CellExecutor, "_execute_r_cell", fake_execute_r)
        import asyncio

        result = asyncio.run(
            executor._materialize_cell(
                "c1",
                "x <- 1",
                timeout_seconds=30.0,
                start_time=0.0,
                materialize_upstreams=False,
                use_cache=False,
            )
        )
        assert calls == ["c1"]
        assert result.success is True


# ---------------------------------------------------------------------------
# ``_run_r_harness`` failure shapes
# ---------------------------------------------------------------------------


class TestRunRHarnessMissingRscript:
    """Rscript not on PATH → friendly error envelope, no subprocess attempt."""

    @pytest.mark.asyncio
    async def test_returns_failure_envelope_when_rscript_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(shutil, "which", lambda name: None)
        _, session = _make_r_notebook(tmp_path, cells=[("c1", None, "x <- 1")])
        executor = CellExecutor(session)

        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text("{}")

        result = await executor._run_r_harness(manifest_path, timeout_seconds=30.0)

        assert result["success"] is False
        assert "Rscript" in result["error"]
        assert result["variables"] == {}


# ---------------------------------------------------------------------------
# Integration — real Rscript
# ---------------------------------------------------------------------------


@rscript_available
class TestExecuteSimpleRCell:
    """Drive a single R cell through ``execute_cell``."""

    @pytest.mark.asyncio
    async def test_simple_assignment_emits_json_artifact(self, tmp_path):
        _, session = _make_r_notebook(tmp_path, cells=[("c1", None, "x <- 1 + 2")])
        executor = CellExecutor(session)

        result = await executor.execute_cell("c1", "x <- 1 + 2")

        assert result.success is True, result.error
        assert "x" in result.outputs
        assert result.outputs["x"]["content_type"] == "json/object"

    @pytest.mark.asyncio
    async def test_dataframe_emits_arrow_artifact(self, tmp_path):
        source = "df <- data.frame(id = 1:3, value = c(10.5, 20.5, 30.5))"
        _, session = _make_r_notebook(tmp_path, cells=[("c1", None, source)])
        executor = CellExecutor(session)

        result = await executor.execute_cell("c1", source)

        assert result.success is True, result.error
        assert "df" in result.outputs
        assert result.outputs["df"]["content_type"] == "arrow/ipc"
        assert result.outputs["df"]["rows"] == 3
        assert result.outputs["df"]["columns"] == 2

    @pytest.mark.asyncio
    async def test_stdout_is_captured(self, tmp_path):
        """``cat()`` to stdout surfaces in the result envelope."""
        source = 'cat("hello from R\\n")\nresult <- 1'
        _, session = _make_r_notebook(tmp_path, cells=[("c1", None, source)])
        executor = CellExecutor(session)

        result = await executor.execute_cell("c1", source)

        assert result.success is True, result.error
        assert "hello from R" in result.stdout

    @pytest.mark.asyncio
    async def test_runtime_error_surfaces_as_failure(self, tmp_path):
        """An R-side ``stop()`` is a non-success with error text."""
        source = 'stop("kaboom")'
        _, session = _make_r_notebook(tmp_path, cells=[("c1", None, source)])
        executor = CellExecutor(session)

        result = await executor.execute_cell("c1", source)

        assert result.success is False
        assert "kaboom" in (result.error or "")


@rscript_available
class TestHarnessVariableSelection:
    """The harness picks rebinds and mutations, not just brand-new names.

    Regression for the post-#57 review: ``setdiff(post_names, pre_names)``
    dropped rebinds (``df <- transform(df, ...)``) because ``df`` was in
    ``pre_names``, leaving the downstream cell consuming the stale
    upstream artifact. The harness now mirrors ``harness.py``'s three-way
    rule (new / value-changed / mutation_defines).
    """

    def _drive_harness(self, manifest_path: Path) -> dict:
        """Run ``harness.R`` and parse the result envelope."""
        import json
        import subprocess

        harness = Path("src/strata/notebook/languages/r/harness.R").resolve()
        subprocess.run(["Rscript", str(harness), str(manifest_path)], check=True)
        return json.loads((manifest_path.parent / "harness-result.json").read_text())

    def _seed_arrow_input(self, output_dir: Path, name: str) -> None:
        """Write a small Arrow IPC stream so the harness can ingest it."""
        import pyarrow as pa

        tbl = pa.table({"a": [1, 2, 3], "b": [10, 20, 30]})
        with pa.ipc.new_stream(output_dir / f"{name}.arrow", tbl.schema) as w:
            w.write_table(tbl)

    def test_rebound_upstream_is_serialized(self, tmp_path):
        """Reassigning an upstream variable must surface as a new artifact."""
        import json

        self._seed_arrow_input(tmp_path, "df")
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "source": "df <- transform(df, c = a * 100)",
                    "inputs": {
                        "df": {
                            "content_type": "arrow/ipc",
                            "file": "df.arrow",
                            "uri": "t://0",
                        }
                    },
                    "output_dir": str(tmp_path),
                    "mounts": {},
                    "env": {},
                    "mutation_defines": [],
                }
            )
        )

        result = self._drive_harness(manifest_path)

        assert result["success"] is True
        assert "df" in result["variables"], (
            "Rebound upstream variable was dropped — setdiff(post, pre) "
            "regression. The three-way emit rule should have caught the "
            "value change via identical()."
        )
        assert result["variables"]["df"]["content_type"] == "arrow/ipc"
        assert result["variables"]["df"]["columns"] == 3

    def test_rds_fallback_emits_for_r_only_values(self, tmp_path):
        """Non-Arrow / non-JSON values are emitted as ``.rds`` with the R-only tag."""
        import json

        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "source": ("model <- lm(b ~ a, data = data.frame(a = 1:3, b = c(2, 4, 6)))"),
                    "inputs": {},
                    "output_dir": str(tmp_path),
                    "mounts": {},
                    "env": {},
                    "mutation_defines": [],
                }
            )
        )

        result = self._drive_harness(manifest_path)

        assert result["success"] is True
        assert "model" in result["variables"]
        payload = result["variables"]["model"]
        assert payload["content_type"] == "application/x-r-rds"
        assert payload["file"] == "model.rds"
        assert payload.get("r_only") is True

    def test_mutation_defines_force_serialization(self, tmp_path):
        """A name listed in ``mutation_defines`` is always emitted.

        Even when ``identical(pre, post)`` still holds — e.g. R's
        copy-on-modify means a column assignment on a data.frame
        produces a structurally equivalent object that may still
        compare ``identical`` if no actual values changed. The
        analyzer's mutation_defines list short-circuits the check.
        """
        import json

        self._seed_arrow_input(tmp_path, "df")
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    # No-op assignment: rebinding the same value to itself.
                    # ``identical(pre, post)`` is True, so without
                    # mutation_defines the binding would be dropped.
                    "source": "df <- df",
                    "inputs": {
                        "df": {
                            "content_type": "arrow/ipc",
                            "file": "df.arrow",
                            "uri": "t://0",
                        }
                    },
                    "output_dir": str(tmp_path),
                    "mounts": {},
                    "env": {},
                    "mutation_defines": ["df"],
                }
            )
        )

        result = self._drive_harness(manifest_path)

        assert result["success"] is True
        assert "df" in result["variables"]


class TestStoreOutputsRdsExtension:
    """``_store_outputs`` recognizes ``.rds`` so R-only artifacts persist.

    Regression for the post-#57 review: the search list previously
    contained only ``.arrow/.json/.pickle/.{cell_,}module.json/.cell_instance.pickle``
    — so an R cell that produced a tagged ``.rds`` artifact (the only
    way to round-trip an arbitrary R object) made the parent log
    "no output file" and report ``stored_ok = False``.
    """

    def test_rds_file_is_persisted_with_r_rds_content_type(self, tmp_path):
        """Drop a ``model.rds`` into the output dir and assert it persists."""
        import json as _json

        from strata.notebook.dag import NotebookDag

        _, session = _make_r_notebook(
            tmp_path,
            cells=[("c1", None, "model <- 1")],
        )
        # _store_outputs only iterates ``consumed_variables[cell_id]``;
        # stub the DAG to declare ``model`` as a downstream consumer.
        session.dag = NotebookDag(consumed_variables={"c1": {"model"}})

        executor = CellExecutor(session)

        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        (output_dir / "model.rds").write_bytes(b"\x1f\x8b\x08\x00fakerds")

        stored = executor._store_outputs(
            "c1",
            output_dir,
            provenance_hash="prov" + "0" * 60,
            input_hashes=[],
        )

        assert stored is True, (
            "_store_outputs missed model.rds — the .rds extension must be "
            "in the search list (review feedback on #57)."
        )

        cell = session.notebook_state.get_cell("c1")
        assert "model" in cell.artifact_uris

        artifact_mgr = session.get_artifact_manager()
        canonical_id = f"nb_{session.notebook_state.id}_cell_c1_var_model"
        artifact = artifact_mgr.artifact_store.get_latest_version(canonical_id)
        assert artifact is not None
        spec = _json.loads(artifact.transform_spec)
        assert spec["params"]["content_type"] == "application/x-r-rds"
