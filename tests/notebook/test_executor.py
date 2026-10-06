"""Tests for the cell executor."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.models import (
    MountMode,
    MountSpec,
    WorkerBackendType,
    WorkerSpec,
)
from strata.notebook.parser import parse_notebook
from strata.notebook.pool import WarmProcessPool
from strata.notebook.session import NotebookSession

_MINIMAL_PNG_LITERAL = (
    'b"\\x89PNG\\r\\n\\x1a\\n\\x00\\x00\\x00\\rIHDR\\x00\\x00\\x00\\x01\\x00\\x00\\x00\\x01'
    "\\x08\\x04\\x00\\x00\\x00\\xb5\\x1c\\x0c\\x02\\x00\\x00\\x00\\x0bIDATx\\xdac\\xfc\\xff"
    '\\x1f\\x00\\x03\\x03\\x02\\x00\\xef\\x9b\\xe0M\\x00\\x00\\x00\\x00IEND\\xaeB`\\x82"'
)
_MARKDOWN_LITERAL = '"# Title\\n\\nA **markdown** cell."'


@pytest.fixture
def sample_notebook(tmp_path):
    """A sample notebook session."""
    from strata.notebook.writer import add_cell_to_notebook, create_notebook

    notebook_dir = create_notebook(tmp_path, "Test Notebook")

    add_cell_to_notebook(notebook_dir, "cell1", None)
    add_cell_to_notebook(notebook_dir, "cell2", "cell1")

    notebook_state = parse_notebook(notebook_dir)
    session = NotebookSession(notebook_state, notebook_dir)
    session.refresh_environment_runtime()

    return session


class TestRevertHitsTheCache:
    """Reverting a cell edit must not recompute a result the store already holds.

    Provenance is content-addressed, so after P1 -> P2 -> P1 the store holds P1's result.
    """

    @pytest.mark.asyncio
    async def test_reverting_an_edit_reuses_the_original_result(self, sample_notebook):
        cell1 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell2 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell2")
        # cell2 consumes x, so cell1's output is stored as a canonical artifact.
        cell2.source = "y = x + 1"
        executor = CellExecutor(sample_notebook)

        cell1.source = "x = 1"
        sample_notebook.re_analyze_cell("cell1")
        sample_notebook.re_analyze_cell("cell2")
        first = await executor.execute_cell("cell1", "x = 1")
        assert first.cache_hit is False

        cell1.source = "x = 2"
        sample_notebook.re_analyze_cell("cell1")
        edited = await executor.execute_cell("cell1", "x = 2")
        assert edited.cache_hit is False

        cell1.source = "x = 1"
        sample_notebook.re_analyze_cell("cell1")
        reverted = await executor.execute_cell("cell1", "x = 1")

        assert reverted.cache_hit is True, "revert re-executed a result already in the store"

        # A downstream cell resolves inputs through the *latest* version, so serving
        # the cache from an older version without re-pointing latest would feed it the
        # edited value. y must be 2 (from x = 1), never 3.
        sample_notebook.re_analyze_cell("cell2")
        downstream = await executor.execute_cell("cell2", "y = x + 1")
        assert downstream.success is True
        assert downstream.outputs["y"]["preview"] == 2

        # The result reports the version it actually made current, not the
        # superseded row the provenance lookup happened to land on.
        store = sample_notebook.artifact_manager.artifact_store
        x_id = f"nb_{sample_notebook.notebook_state.id}_cell_cell1_var_x"
        latest = store.get_latest_version(x_id)
        assert reverted.artifact_uri == f"strata://artifact/{x_id}@v={latest.version}"
        assert latest.state == "ready"

    @pytest.mark.asyncio
    async def test_revert_keeps_the_display_output(self, sample_notebook):
        cell1 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell2 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell2")
        cell2.source = "y = x + 1"
        executor = CellExecutor(sample_notebook)

        src1 = 'x = 1\ndisplay(Markdown("# One"))'
        src2 = 'x = 2\ndisplay(Markdown("# Two"))'

        cell1.source = src1
        sample_notebook.re_analyze_cell("cell1")
        sample_notebook.re_analyze_cell("cell2")
        first = await executor.execute_cell("cell1", src1)
        assert first.display_outputs, "setup: first run should have a display output"

        cell1.source = src2
        sample_notebook.re_analyze_cell("cell1")
        await executor.execute_cell("cell1", src2)

        cell1.source = src1
        sample_notebook.re_analyze_cell("cell1")
        reverted = await executor.execute_cell("cell1", src1)

        assert reverted.display_outputs, "revert lost the display output"

    @pytest.mark.asyncio
    async def test_a_cell_that_must_run_anyway_promotes_nothing(self, sample_notebook):
        """One variable promotable and another not means the cell runs and nothing is promoted.

        A promoted artifact would disagree with the rest of the cell until the run overwrote it.
        """
        cell1 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell2 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell2")
        cell2.source = "z = a + b"
        executor = CellExecutor(sample_notebook)
        store = sample_notebook.artifact_manager.artifact_store
        notebook_id = sample_notebook.notebook_state.id
        a_id = f"nb_{notebook_id}_cell_cell1_var_a"
        b_id = f"nb_{notebook_id}_cell_cell1_var_b"

        cell1.source = "a = 1\nb = 10"
        sample_notebook.re_analyze_cell("cell1")
        sample_notebook.re_analyze_cell("cell2")
        await executor.execute_cell("cell1", "a = 1\nb = 10")

        cell1.source = "a = 2\nb = 20"
        sample_notebook.re_analyze_cell("cell1")
        await executor.execute_cell("cell1", "a = 2\nb = 20")

        # b's original result is gone, so the revert cannot be served entirely
        # from the store.
        assert store.delete_artifact(b_id, 1) is True

        cell1.source = "a = 1\nb = 10"
        sample_notebook.re_analyze_cell("cell1")
        reverted = await executor.execute_cell("cell1", "a = 1\nb = 10")

        assert reverted.cache_hit is False
        # a went v1, v2, then the re-execution's v3. A promotion would have
        # slipped an extra version in before that.
        assert store.get_latest_version(a_id).version == 3
        # Both variables end up current under the reverted provenance, via the
        # run rather than via a promotion.
        assert reverted.outputs["a"]["preview"] == 1
        assert reverted.outputs["b"]["preview"] == 10


class TestManifestTablesDuplicates:
    """Duplicate @table names are rejected before manifest and provenance.

    The name-keyed snapshot map would collapse them: one wins namespace injection, and an
    unresolved duplicate can borrow a resolved one's snapshot.
    """

    def test_duplicate_table_name_rejected(self, sample_notebook):
        from strata.notebook.models import TableSpec

        executor = CellExecutor(sample_notebook)
        specs = [
            TableSpec(name="trips", uri="file:///wh#a.trips"),
            TableSpec(name="trips", uri="file:///wh#b.trips"),
        ]
        with pytest.raises(RuntimeError, match="duplicate @table"):
            executor._manifest_tables(specs, {"trips": 123})


class TestManifestTablesEmpty:
    def test_a_table_never_written_is_injected_with_no_snapshot(self, sample_notebook):
        from strata.notebook.models import TableSpec

        executor = CellExecutor(sample_notebook)
        specs = [TableSpec(name="trips", uri="file:///wh#a.trips")]

        tables = executor._manifest_tables(specs, {"trips": None})

        assert tables == {"trips": {"uri": "file:///wh#a.trips", "snapshot_id": None}}
        with pytest.raises(RuntimeError, match="could not resolve a snapshot"):
            executor._manifest_tables(specs, {})


class TestCellExecutor:
    @pytest.mark.asyncio
    async def test_execute_simple_assignment(self, sample_notebook):
        executor = CellExecutor(sample_notebook)

        source = "x = 1 + 1"
        result = await executor.execute_cell("cell1", source)

        assert result.success is True
        assert result.cell_id == "cell1"
        assert result.error is None
        assert "x" in result.outputs
        assert result.outputs["x"]["content_type"] == "json/object"
        assert result.outputs["x"]["preview"] == 2

    @pytest.mark.asyncio
    async def test_execute_with_print(self, sample_notebook):
        executor = CellExecutor(sample_notebook)

        source = 'print("Hello, world!")\ny = 42'
        result = await executor.execute_cell("cell1", source)

        assert result.success is True
        assert "Hello, world!" in result.stdout
        assert "y" in result.outputs

    @pytest.mark.asyncio
    async def test_cache_hit_preserves_persisted_console(self, sample_notebook):
        """A cache hit carries no fresh stdout; it must not unlink the console file the first run
        persisted.
        """
        executor = CellExecutor(sample_notebook)
        console_file = sample_notebook.path / ".strata" / "console" / "cell1.json"

        # A leaf display output makes the cell cache-hittable on rerun; the
        # print gives it console to persist.
        source = (
            'print("recoverable output")\n'
            "class D:\n"
            "    def _repr_markdown_(self):\n"
            '        return "# done"\n'
            "D()\n"
        )
        first = await executor.execute_cell("cell1", source)
        assert first.success is True
        assert first.cache_hit is False
        assert console_file.is_file()
        assert "recoverable output" in console_file.read_text()

        second = await executor.execute_cell("cell1", source)
        assert second.success is True
        assert second.cache_hit is True
        assert console_file.is_file()
        assert "recoverable output" in console_file.read_text()

    @pytest.mark.asyncio
    async def test_silent_leaf_hits_the_cache_on_rerun(self, sample_notebook):
        """A leaf with nothing consumed and nothing printed still ran; an unchanged re-run is a
        hit, not another subprocess.
        """
        executor = CellExecutor(sample_notebook)
        # cell2 is empty, so nothing reads x: no variable artifact, no display, no console.
        source = "import os\nx = 1\n"

        first = await executor.execute_cell("cell1", source)
        assert first.success is True
        assert first.cache_hit is False

        second = await executor.execute_cell("cell1", source)
        assert second.success is True
        assert second.cache_hit is True
        assert second.stdout == ""
        assert second.stderr == ""

    @pytest.mark.asyncio
    async def test_execute_with_error(self, sample_notebook):
        executor = CellExecutor(sample_notebook)

        source = "z = 1 / 0"
        result = await executor.execute_cell("cell1", source)

        assert result.success is False
        assert result.error is not None
        assert "ZeroDivisionError" in result.error or "division" in result.error

    @pytest.mark.asyncio
    async def test_execute_dataframe(self, sample_notebook):
        executor = CellExecutor(sample_notebook)

        # A dict, since pandas may be absent from the test venv.
        source = 'df = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}'
        result = await executor.execute_cell("cell1", source)

        assert result.success is True
        assert "df" in result.outputs
        assert result.outputs["df"]["content_type"] == "json/object"
        assert result.outputs["df"]["preview"] == {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}

    @pytest.mark.asyncio
    async def test_execute_multiple_outputs(self, sample_notebook):
        executor = CellExecutor(sample_notebook)

        source = """
x = 10
y = "hello"
z = [1, 2, 3]
"""
        result = await executor.execute_cell("cell1", source)

        assert result.success is True
        assert set(result.outputs.keys()) == {"x", "y", "z"}
        assert result.outputs["x"]["preview"] == 10
        assert result.outputs["y"]["preview"] == "hello"
        assert result.outputs["z"]["preview"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_shared_mutable_outputs_warns_through_real_execution(self, sample_notebook):
        """Two outputs sharing a mutable object warn through real subprocess execution into
        ``result.mutation_warnings``, which `strata run` prints.
        """
        executor = CellExecutor(sample_notebook)

        # params and trainer share the same list object → decouple once stored.
        source = "params = [1.0, 2.0, 3.0]\ntrainer = {'params': params}"
        result = await executor.execute_cell("cell1", source)

        assert result.success is True
        messages = [w["message"] for w in result.mutation_warnings]
        assert any("share a mutable" in m for m in messages), messages

    @pytest.mark.asyncio
    async def test_execute_dict_output(self, sample_notebook):
        executor = CellExecutor(sample_notebook)

        source = 'data = {"count": 42, "names": ["Alice", "Bob"]}'
        result = await executor.execute_cell("cell1", source)

        assert result.success is True
        assert "data" in result.outputs
        assert result.outputs["data"]["content_type"] == "json/object"

    @pytest.mark.asyncio
    async def test_execute_png_display_output_is_cached_for_leaf_cells(self, sample_notebook):
        """Leaf display outputs should persist as display artifacts and cache-hit on rerun."""
        executor = CellExecutor(sample_notebook)
        source = f"""
class Display:
    def _repr_png_(self):
        return {_MINIMAL_PNG_LITERAL}

Display()
"""

        first = await executor.execute_cell("cell1", source)

        assert first.success is True
        assert first.cache_hit is False
        assert first.display_output is not None
        assert first.display_output["content_type"] == "image/png"
        assert first.display_output["artifact_uri"].startswith("strata://artifact/")
        assert first.outputs["_"]["content_type"] == "image/png"

        second = await executor.execute_cell("cell1", source)

        assert second.success is True
        assert second.cache_hit is True
        assert second.display_output is not None
        assert second.display_output["content_type"] == "image/png"
        assert second.display_output["artifact_uri"].startswith("strata://artifact/")

    @pytest.mark.asyncio
    async def test_execute_stdout_is_cached_for_leaf_cells(self, sample_notebook):
        """A print-only leaf (no display, no consumer) caches by provenance and replays its stdout
        on an unchanged re-run: the agent scratchpad case.
        """
        executor = CellExecutor(sample_notebook)
        source = "print('hello from a leaf cell')\n"

        first = await executor.execute_cell("cell1", source)
        assert first.success is True
        assert first.cache_hit is False
        assert "hello from a leaf cell" in first.stdout

        second = await executor.execute_cell("cell1", source)
        assert second.success is True
        assert second.cache_hit is True
        assert second.execution_method == "cached"
        # stdout is replayed from the provenance-keyed console cache.
        assert "hello from a leaf cell" in second.stdout

    @pytest.mark.asyncio
    async def test_nocache_annotation_always_reexecutes(self, sample_notebook):
        """`# @nocache` re-runs a cell every time, even with source, inputs and env unchanged."""
        executor = CellExecutor(sample_notebook)
        source = "# @nocache\nprint('side effect')\n"

        first = await executor.execute_cell("cell1", source)
        assert first.success is True
        assert first.cache_hit is False

        second = await executor.execute_cell("cell1", source)
        assert second.success is True
        assert second.cache_hit is False  # never cached

    @staticmethod
    def _counting_source(counter, *, prefix: str = "") -> str:
        """A leaf cell that bumps a file counter and displays the new count.

        A replayed display looks like a real run; the file says how many times it really ran.
        """
        return (
            f"{prefix}from pathlib import Path\n"
            f"_p = Path({str(counter)!r})\n"
            "_p.write_text(str(int(_p.read_text()) + 1) if _p.exists() else '1')\n"
            "int(_p.read_text())\n"
        )

    @pytest.mark.asyncio
    async def test_nocache_leaf_reexecutes_its_displayed_value(self, sample_notebook, tmp_path):
        """`# @nocache` must also gate the leaf display-replay path.

        A leaf's display is resolved before the cache decision, so an ungated replay would hand
        back the first run's value forever; the print-only test above cannot catch that.
        """
        executor = CellExecutor(sample_notebook)
        counter = tmp_path / "nocache_counter.txt"
        source = self._counting_source(counter, prefix="# @nocache\n")

        first = await executor.execute_cell("cell1", source)
        assert first.success is True
        assert first.cache_hit is False
        assert first.display_output["preview"] == 1

        second = await executor.execute_cell("cell1", source)
        assert second.success is True
        assert second.cache_hit is False
        assert second.display_output["preview"] == 2
        assert counter.read_text() == "2"

    @pytest.mark.asyncio
    async def test_force_and_rerun_reexecute_a_displayed_leaf(self, sample_notebook, tmp_path):
        """Force and rerun turn the cache off; a displayed leaf is not exempt."""
        executor = CellExecutor(sample_notebook)
        counter = tmp_path / "forced_counter.txt"
        source = self._counting_source(counter)

        first = await executor.execute_cell("cell1", source)
        assert first.display_output["preview"] == 1

        # Normal mode still replays -- that is the caching payoff.
        cached = await executor.execute_cell("cell1", source)
        assert cached.cache_hit is True
        assert cached.display_output["preview"] == 1

        forced = await executor.execute_cell_force("cell1", source)
        assert forced.cache_hit is False
        assert forced.display_output["preview"] == 2

        reran = await executor.execute_cell_rerun("cell1", source)
        assert reran.cache_hit is False
        assert reran.display_output["preview"] == 3
        assert counter.read_text() == "3"

    @pytest.mark.asyncio
    async def test_execute_markdown_display_output_is_cached_for_leaf_cells(self, sample_notebook):
        """Markdown display outputs should persist as display artifacts and cache-hit on rerun."""
        executor = CellExecutor(sample_notebook)
        source = f"""
class Display:
    def _repr_markdown_(self):
        return {_MARKDOWN_LITERAL}

Display()
"""

        first = await executor.execute_cell("cell1", source)

        assert first.success is True
        assert first.cache_hit is False
        assert first.display_output is not None
        assert first.display_output["content_type"] == "text/markdown"
        assert first.display_output["artifact_uri"].startswith("strata://artifact/")
        assert first.display_output["markdown_text"] == "# Title\n\nA **markdown** cell."

        second = await executor.execute_cell("cell1", source)

        assert second.success is True
        assert second.cache_hit is True
        assert second.display_output is not None
        assert second.display_output["content_type"] == "text/markdown"
        assert second.display_output["artifact_uri"].startswith("strata://artifact/")
        assert second.display_output["markdown_text"] == "# Title\n\nA **markdown** cell."

    @pytest.mark.asyncio
    async def test_execute_display_side_effect_png_output_is_cached(self, sample_notebook):
        """display(...) side effects should flow through the primary display cache path."""
        executor = CellExecutor(sample_notebook)
        source = f"""
class Display:
    def _repr_png_(self):
        return {_MINIMAL_PNG_LITERAL}

display(Display())
"""

        first = await executor.execute_cell("cell1", source)

        assert first.success is True
        assert first.cache_hit is False
        assert first.display_output is not None
        assert first.display_output["content_type"] == "image/png"
        assert first.outputs["_"]["content_type"] == "image/png"

        second = await executor.execute_cell("cell1", source)

        assert second.success is True
        assert second.cache_hit is True
        assert second.display_output is not None
        assert second.display_output["content_type"] == "image/png"

    @pytest.mark.asyncio
    async def test_execute_display_side_effect_markdown_output_is_cached(self, sample_notebook):
        """display(Markdown(...)) should persist and cache-hit like explicit returned outputs."""
        executor = CellExecutor(sample_notebook)
        source = """
display(Markdown("# Side effect\\n\\nCaptured."))
"""

        first = await executor.execute_cell("cell1", source)

        assert first.success is True
        assert first.cache_hit is False
        assert first.display_output is not None
        assert first.display_output["content_type"] == "text/markdown"
        assert first.display_output["markdown_text"] == "# Side effect\n\nCaptured."

        second = await executor.execute_cell("cell1", source)

        assert second.success is True
        assert second.cache_hit is True
        assert second.display_output is not None
        assert second.display_output["content_type"] == "text/markdown"
        assert second.display_output["markdown_text"] == "# Side effect\n\nCaptured."

    @pytest.mark.asyncio
    async def test_execute_last_expression_overrides_earlier_display_side_effect(
        self, sample_notebook
    ):
        """The last visible result should remain the legacy primary display shim."""
        executor = CellExecutor(sample_notebook)
        source = """
display(Markdown("# Earlier"))
42
"""

        result = await executor.execute_cell("cell1", source)

        assert result.success is True
        assert len(result.display_outputs) == 2
        assert result.display_outputs[0]["content_type"] == "text/markdown"
        assert result.display_outputs[0]["markdown_text"] == "# Earlier"
        assert result.display_outputs[1]["content_type"] == "json/object"
        assert result.display_outputs[1]["preview"] == 42
        assert result.display_output is not None
        assert result.display_output["content_type"] == "json/object"
        assert result.display_output["preview"] == 42

    @pytest.mark.asyncio
    async def test_execute_multiple_display_outputs_are_cached_in_order(self, sample_notebook):
        """Ordered display outputs should persist and survive cache hits."""
        executor = CellExecutor(sample_notebook)
        source = """
display(Markdown("# First"))
42
"""

        first = await executor.execute_cell("cell1", source)

        assert first.success is True
        assert first.cache_hit is False
        assert len(first.display_outputs) == 2
        assert first.display_outputs[0]["content_type"] == "text/markdown"
        assert first.display_outputs[0]["artifact_uri"].startswith("strata://artifact/")
        assert first.display_outputs[1]["content_type"] == "json/object"
        assert first.display_outputs[1]["artifact_uri"].startswith("strata://artifact/")
        assert first.display_output is not None
        assert first.display_output["content_type"] == "json/object"
        assert first.display_output["preview"] == 42

        second = await executor.execute_cell("cell1", source)

        assert second.success is True
        assert second.cache_hit is True
        assert len(second.display_outputs) == 2
        assert second.display_outputs[0]["content_type"] == "text/markdown"
        assert second.display_outputs[1]["content_type"] == "json/object"
        assert second.display_output is not None
        assert second.display_output["preview"] == 42

    @pytest.mark.asyncio
    async def test_execute_ignores_private_vars(self, sample_notebook):
        executor = CellExecutor(sample_notebook)

        source = """
public = 1
_private = 2
"""
        result = await executor.execute_cell("cell1", source)

        assert result.success is True
        assert "public" in result.outputs
        assert "_private" not in result.outputs

    @pytest.mark.asyncio
    async def test_execute_empty_cell(self, sample_notebook):
        executor = CellExecutor(sample_notebook)

        source = "# Just a comment"
        result = await executor.execute_cell("cell1", source)

        assert result.success is True
        assert len(result.outputs) == 0

    @pytest.mark.asyncio
    async def test_execute_with_import(self, sample_notebook):
        executor = CellExecutor(sample_notebook)

        source = """
import math
result = math.pi
"""
        result = await executor.execute_cell("cell1", source)

        assert result.success is True
        assert "result" in result.outputs
        assert abs(result.outputs["result"]["preview"] - 3.14159) < 0.01

    @pytest.mark.asyncio
    async def test_execute_with_stderr(self, sample_notebook):
        executor = CellExecutor(sample_notebook)

        source = """
import sys
print("error message", file=sys.stderr)
x = 1
"""
        result = await executor.execute_cell("cell1", source)

        assert result.success is True
        assert "error message" in result.stderr

    @pytest.mark.asyncio
    async def test_execute_surfaces_mount_resolution_error(self, sample_notebook):
        """Mount resolution failures should be surfaced, not silently ignored."""
        executor = CellExecutor(sample_notebook)
        cell = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell.mounts = [
            MountSpec(
                name="raw_data",
                uri="file:///definitely/not/a/real/path",
                mode=MountMode.READ_ONLY,
            )
        ]

        result = await executor.execute_cell("cell1", "x = raw_data")

        assert result.success is False
        assert result.error is not None
        assert "Local mount 'raw_data' path does not exist" in result.error

    @pytest.mark.asyncio
    async def test_execute_fails_when_rw_sync_back_fails(
        self,
        sample_notebook,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        """RW mount sync-back failures should mark the execution failed."""
        executor = CellExecutor(sample_notebook)
        cell = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell.mounts = [
            MountSpec(
                name="scratch",
                uri=f"file://{tmp_path / 'scratch'}",
                mode=MountMode.READ_WRITE,
            )
        ]

        async def _boom(*args, **kwargs):
            raise RuntimeError("sync failed")

        monkeypatch.setattr(executor._mount_resolver, "sync_back", _boom)

        result = await executor.execute_cell(
            "cell1",
            '(scratch / "result.txt").write_text("ok")',
        )

        assert result.success is False
        assert result.error is not None
        assert "failed to sync read-write mounts: sync failed" in result.error

    @pytest.mark.asyncio
    async def test_execute_applies_env_annotations(self, sample_notebook):
        """@env annotations should be visible inside cell execution."""
        executor = CellExecutor(sample_notebook)

        source = """
# @env NOTEBOOK_TOKEN=secret-value
import os
token = os.getenv("NOTEBOOK_TOKEN")
"""
        result = await executor.execute_cell("cell1", source)

        assert result.success is True
        assert result.outputs["token"]["preview"] == "secret-value"

    @pytest.mark.asyncio
    async def test_execute_applies_persisted_env_defaults(self, sample_notebook):
        """Persisted env defaults should be visible inside cell execution."""
        executor = CellExecutor(sample_notebook)
        cell = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        sample_notebook.notebook_state.env = {"NOTEBOOK_TOKEN": "saved-default"}
        cell.env = {"NOTEBOOK_TOKEN": "saved-default"}

        source = """
import os
token = os.getenv("NOTEBOOK_TOKEN")
"""
        result = await executor.execute_cell("cell1", source)

        assert result.success is True
        assert result.outputs["token"]["preview"] == "saved-default"

    @pytest.mark.asyncio
    async def test_blanked_secret_on_disk_does_not_mask_server_env(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        """A sensitive key blanked in notebook.toml leaves the server's value visible.

        An empty non-sensitive value is still an intentional override.
        """
        from strata.notebook.writer import (
            add_cell_to_notebook,
            create_notebook,
            update_notebook_env,
        )

        monkeypatch.setenv("STRATA_TEST_API_KEY", "server-secret")
        monkeypatch.setenv("STRATA_TEST_PLAIN", "server-plain")
        notebook_dir = create_notebook(tmp_path, "Env Notebook")
        add_cell_to_notebook(notebook_dir, "cell1", None)
        add_cell_to_notebook(notebook_dir, "cell2", "cell1")
        update_notebook_env(
            notebook_dir,
            {"STRATA_TEST_API_KEY": "typed-secret", "STRATA_TEST_PLAIN": "", "OTHER": "x"},
        )
        toml_text = (notebook_dir / "notebook.toml").read_text()
        assert 'STRATA_TEST_API_KEY = ""' in toml_text

        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        session.refresh_environment_runtime()
        # The key name stays for re-entry in the UI.
        assert session.notebook_state.env["STRATA_TEST_API_KEY"] == ""

        source = """
import os
secret = os.environ.get("STRATA_TEST_API_KEY")
plain = os.environ.get("STRATA_TEST_PLAIN")
"""
        cell = next(c for c in session.notebook_state.cells if c.id == "cell1")
        cell.source = source
        # A consumer, so cell1's outputs are stored and staleness can report ready.
        consumer = next(c for c in session.notebook_state.cells if c.id == "cell2")
        consumer.source = "pair = (secret, plain)"
        session.re_analyze_cell("cell1")
        session.re_analyze_cell("cell2")
        result = await CellExecutor(session).execute_cell("cell1", source)

        assert result.success is True, result.error
        assert result.outputs["secret"]["preview"] == "server-secret"
        assert result.outputs["plain"]["preview"] == ""
        # Staleness drops the blanked key too, so the cell is not stale forever.
        assert session.compute_staleness()["cell1"].status.value == "ready"

    @pytest.mark.asyncio
    async def test_execute_uses_timeout_annotation(
        self,
        sample_notebook,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """@timeout annotations should override the executor timeout."""
        executor = CellExecutor(sample_notebook)
        observed: list[float] = []

        async def _fake_run_harness(
            manifest_path: Path,
            venv_python: Path,
            timeout_seconds: float,
        ) -> dict[str, object]:
            del manifest_path, venv_python
            observed.append(timeout_seconds)
            return {
                "success": True,
                "variables": {},
                "stdout": "",
                "stderr": "",
                "mutation_warnings": [],
            }

        monkeypatch.setattr(executor, "_run_harness", _fake_run_harness)

        result = await executor.execute_cell(
            "cell1",
            "# @timeout 1.5\nx = 1",
            timeout_seconds=30,
        )

        assert result.success is True
        assert observed == [1.5]

    @pytest.mark.asyncio
    async def test_execute_uses_persisted_timeout_override(
        self,
        sample_notebook,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Persisted timeout defaults should affect execution timeout."""
        executor = CellExecutor(sample_notebook)
        observed: list[float] = []
        cell = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        sample_notebook.notebook_state.timeout = 6.0
        cell.timeout = 2.5
        cell.timeout_override = 2.5

        async def _fake_run_harness(
            manifest_path: Path,
            venv_python: Path,
            timeout_seconds: float,
        ) -> dict[str, object]:
            del manifest_path, venv_python
            observed.append(timeout_seconds)
            return {
                "success": True,
                "variables": {},
                "stdout": "",
                "stderr": "",
                "mutation_warnings": [],
            }

        monkeypatch.setattr(executor, "_run_harness", _fake_run_harness)

        result = await executor.execute_cell("cell1", "x = 1", timeout_seconds=30)

        assert result.success is True
        assert observed == [2.5]

    @pytest.mark.asyncio
    async def test_execute_rejects_unimplemented_worker_annotation(
        self,
        sample_notebook,
    ):
        """An unregistered @worker fails fast."""
        executor = CellExecutor(sample_notebook)

        result = await executor.execute_cell(
            "cell1",
            "# @worker gpu-a100\nx = 1",
        )

        assert result.success is False
        assert result.error == "Execution failed: worker 'gpu-a100' is not implemented yet"

    @pytest.mark.asyncio
    async def test_execute_allows_registered_local_worker_annotation(
        self,
        sample_notebook,
    ):
        """Named local workers should execute through the existing local path."""
        executor = CellExecutor(sample_notebook)
        sample_notebook.notebook_state.workers = [
            WorkerSpec(
                name="cpu-analytics",
                backend=WorkerBackendType.LOCAL,
                runtime_id="python-analytics",
            )
        ]

        result = await executor.execute_cell(
            "cell1",
            "# @worker cpu-analytics\nx = 1",
        )

        assert result.success is True
        assert result.error is None
        assert "x" in result.outputs

    @pytest.mark.asyncio
    async def test_execute_rejects_unimplemented_notebook_worker(
        self,
        sample_notebook,
    ):
        """Persisted notebook worker defaults should affect execution routing."""
        executor = CellExecutor(sample_notebook)
        sample_notebook.notebook_state.worker = "gpu-default"
        cell = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell.worker = "gpu-default"

        result = await executor.execute_cell("cell1", "x = 1")

        assert result.success is False
        assert result.error == "Execution failed: worker 'gpu-default' is not implemented yet"

    @pytest.mark.asyncio
    async def test_execute_rejects_unimplemented_cell_worker_override(
        self,
        sample_notebook,
    ):
        """Persisted cell worker overrides should take precedence over notebook defaults."""
        executor = CellExecutor(sample_notebook)
        sample_notebook.notebook_state.worker = "gpu-default"
        cell = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell.worker = "gpu-override"
        cell.worker_override = "gpu-override"

        result = await executor.execute_cell("cell1", "x = 1")

        assert result.success is False
        assert result.error == "Execution failed: worker 'gpu-override' is not implemented yet"

    @pytest.mark.asyncio
    async def test_execute_rejects_disallowed_service_mode_worker(
        self,
        sample_notebook,
        monkeypatch,
    ):
        """Service mode should reject notebook-local workers outside the server registry."""
        monkeypatch.setattr(
            "strata.server._state",
            SimpleNamespace(
                config=SimpleNamespace(
                    deployment_mode="service",
                    transforms_config={
                        "notebook_workers": [
                            {
                                "name": "gpu-a100",
                                "backend": "executor",
                                "runtime_id": "cuda-12.4",
                                "config": {"url": "embedded://local"},
                            }
                        ]
                    },
                )
            ),
        )
        sample_notebook.notebook_state.workers = [
            WorkerSpec(
                name="gpu-shadow",
                backend=WorkerBackendType.EXECUTOR,
                runtime_id="shadow-runtime",
                config={"url": "embedded://local"},
            )
        ]
        sample_notebook.notebook_state.worker = "gpu-shadow"
        cell = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell.worker = "gpu-shadow"

        result = await CellExecutor(sample_notebook).execute_cell("cell1", "x = 1")

        assert result.success is False
        assert result.error == (
            "Execution failed: Worker 'gpu-shadow' is not allowed in service mode. "
            "Choose a server-managed worker."
        )

    @pytest.mark.asyncio
    async def test_execute_rejects_disabled_service_mode_worker(
        self,
        sample_notebook,
        monkeypatch,
    ):
        """Service mode should reject disabled server-managed workers."""
        monkeypatch.setattr(
            "strata.server._state",
            SimpleNamespace(
                config=SimpleNamespace(
                    deployment_mode="service",
                    transforms_config={
                        "notebook_workers": [
                            {
                                "name": "gpu-a100",
                                "backend": "executor",
                                "runtime_id": "cuda-12.4",
                                "config": {"url": "embedded://local"},
                                "enabled": False,
                            }
                        ]
                    },
                )
            ),
        )
        sample_notebook.notebook_state.worker = "gpu-a100"
        cell = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell.worker = "gpu-a100"

        result = await CellExecutor(sample_notebook).execute_cell("cell1", "x = 1")

        assert result.success is False
        assert result.error == (
            "Execution failed: Worker 'gpu-a100' is disabled by server policy. "
            "Choose an enabled server-managed worker."
        )

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform == "win32", reason="harness users are POSIX")
    async def test_execute_allows_service_mode_server_worker(
        self,
        sample_notebook,
        monkeypatch,
    ):
        """Service mode should resolve executor workers from the server registry.

        An ``embedded://`` worker runs on this host, which service mode refuses
        without a harness user; the current user stands in for one here.
        """
        import pwd

        monkeypatch.setattr(
            "strata.server._state",
            SimpleNamespace(
                config=SimpleNamespace(
                    notebook_harness_user=pwd.getpwuid(os.getuid()).pw_name,
                    deployment_mode="service",
                    transforms_config={
                        "notebook_workers": [
                            {
                                "name": "gpu-a100",
                                "backend": "executor",
                                "runtime_id": "cuda-12.4",
                                "config": {"url": "embedded://local"},
                            }
                        ]
                    },
                )
            ),
        )
        sample_notebook.notebook_state.worker = "gpu-a100"
        cell = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell.worker = "gpu-a100"

        result = await CellExecutor(sample_notebook).execute_cell("cell1", "x = 1")

        assert result.success is True
        assert result.execution_method == "executor"

    @pytest.mark.asyncio
    async def test_worker_runtime_identity_invalidates_cache(
        self,
        sample_notebook,
    ):
        """Changing a local worker runtime identity should invalidate cache."""
        sample_notebook.notebook_state.workers = [
            WorkerSpec(
                name="cpu-analytics",
                backend=WorkerBackendType.LOCAL,
                runtime_id="py311-a",
            )
        ]
        sample_notebook.notebook_state.worker = "cpu-analytics"
        cell1 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell2 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell2")
        cell1.worker = "cpu-analytics"
        cell2.worker = "cpu-analytics"
        cell1.source = "x = 1"
        cell2.source = "y = x + 1"
        sample_notebook.re_analyze_cell("cell1")
        sample_notebook.re_analyze_cell("cell2")

        executor = CellExecutor(sample_notebook)

        first = await executor.execute_cell("cell1", "x = 1")
        second = await executor.execute_cell("cell1", "x = 1")

        assert first.success is True
        assert first.cache_hit is False
        assert second.success is True
        assert second.cache_hit is True

        sample_notebook.notebook_state.workers = [
            WorkerSpec(
                name="cpu-analytics",
                backend=WorkerBackendType.LOCAL,
                runtime_id="py311-b",
            )
        ]

        third = await executor.execute_cell("cell1", "x = 1")

        assert third.success is True
        assert third.cache_hit is False

    @pytest.mark.asyncio
    async def test_execute_supports_embedded_executor_worker(
        self,
        sample_notebook,
    ):
        """Supported executor workers should use the bundle-based executor path."""
        sample_notebook.notebook_state.workers = [
            WorkerSpec(
                name="gpu-embedded",
                backend=WorkerBackendType.EXECUTOR,
                runtime_id="gpu-a100",
                config={"url": "embedded://local"},
            )
        ]
        sample_notebook.notebook_state.worker = "gpu-embedded"
        cell1 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell2 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell2")
        cell1.worker = "gpu-embedded"
        cell2.worker = "gpu-embedded"
        cell1.source = "x = 1"
        cell2.source = "y = x + 1"
        sample_notebook.re_analyze_cell("cell1")
        sample_notebook.re_analyze_cell("cell2")

        executor = CellExecutor(sample_notebook)
        first = await executor.execute_cell("cell1", "x = 1")
        second = await executor.execute_cell("cell1", "x = 1")

        assert first.success is True
        assert first.execution_method == "executor"
        assert first.cache_hit is False
        assert first.remote_worker == "gpu-embedded"
        assert first.remote_transport == "embedded"
        assert first.remote_build_id is None
        assert "x" in first.outputs

        assert second.success is True
        assert second.cache_hit is True
        assert second.remote_worker == "gpu-embedded"
        assert second.remote_transport == "embedded"

    @pytest.mark.asyncio
    async def test_execute_supports_http_executor_worker(
        self,
        sample_notebook,
        notebook_executor_server,
    ):
        """HTTP executor workers should execute through the remote bundle transport."""
        sample_notebook.notebook_state.workers = [
            WorkerSpec(
                name="gpu-http",
                backend=WorkerBackendType.EXECUTOR,
                runtime_id="gpu-http-a100",
                config={"url": notebook_executor_server["execute_url"]},
            )
        ]
        sample_notebook.notebook_state.worker = "gpu-http"
        cell1 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell2 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell2")
        cell1.worker = "gpu-http"
        cell2.worker = "gpu-http"
        cell1.source = "x = 1"
        cell2.source = "y = x + 1"
        sample_notebook.re_analyze_cell("cell1")
        sample_notebook.re_analyze_cell("cell2")

        executor = CellExecutor(sample_notebook)
        first = await executor.execute_cell("cell1", "x = 1")
        second = await executor.execute_cell("cell1", "x = 1")

        assert first.success is True
        assert first.execution_method == "executor"
        assert first.cache_hit is False
        assert "x" in first.outputs

        assert second.success is True
        assert second.cache_hit is True

    @pytest.mark.asyncio
    async def test_execute_supports_http_executor_worker_with_class_instances(
        self,
        tmp_path,
        notebook_executor_server,
    ):
        """HTTP executor workers should preserve exported class instances across cells."""
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        notebook_dir = create_notebook(tmp_path, "http_class_instances")
        add_cell_to_notebook(notebook_dir, "cell1", None)
        add_cell_to_notebook(notebook_dir, "cell2", "cell1")
        add_cell_to_notebook(notebook_dir, "cell3", "cell2")

        write_cell(
            notebook_dir,
            "cell1",
            """
class Person:
    name = "John"
    age = 20

    def __str__(self):
        return f"{self.name}:{self.age}"
""".strip(),
        )
        write_cell(notebook_dir, "cell2", "p = Person()")
        write_cell(notebook_dir, "cell3", "rendered = str(p)")

        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        session.notebook_state.workers = [
            WorkerSpec(
                name="gpu-http",
                backend=WorkerBackendType.EXECUTOR,
                runtime_id="gpu-http-a100",
                config={"url": notebook_executor_server["execute_url"]},
            )
        ]
        session.notebook_state.worker = "gpu-http"
        for cell in session.notebook_state.cells:
            cell.worker = "gpu-http"

        executor = CellExecutor(session)
        first = await executor.execute_cell("cell1", session.notebook_state.cells[0].source)
        second = await executor.execute_cell("cell2", session.notebook_state.cells[1].source)
        third = await executor.execute_cell("cell3", session.notebook_state.cells[2].source)

        assert first.success is True
        assert first.execution_method == "executor"
        assert second.success is True
        assert second.execution_method == "executor"
        assert second.outputs["p"]["content_type"] == "module/cell-instance"
        assert third.success is True
        assert third.execution_method == "executor"
        assert third.outputs["rendered"]["preview"] == "John:20"

    @pytest.mark.asyncio
    async def test_execute_supports_signed_http_executor_worker(
        self,
        sample_notebook,
        notebook_executor_server,
        notebook_build_server,
    ):
        """HTTP executor workers can opt into the build + signed-URL transport."""
        notebook_build_server["config"].transforms_config["notebook_workers"] = [
            {
                "name": "gpu-http-signed",
                "backend": "executor",
                "runtime_id": "gpu-http-signed-a100",
                "config": {
                    "url": notebook_executor_server["execute_url"],
                    "transport": "signed",
                    "strata_url": notebook_build_server["base_url"],
                },
            }
        ]
        sample_notebook.notebook_state.workers = [
            WorkerSpec(
                name="gpu-http-signed",
                backend=WorkerBackendType.EXECUTOR,
                runtime_id="gpu-http-signed-a100",
                config={
                    "url": notebook_executor_server["execute_url"],
                    "transport": "signed",
                    "strata_url": notebook_build_server["base_url"],
                },
            )
        ]
        sample_notebook.notebook_state.worker = "gpu-http-signed"
        cell1 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell2 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell2")
        cell1.worker = "gpu-http-signed"
        cell2.worker = "gpu-http-signed"
        cell1.source = "x = 1"
        cell2.source = "y = x + 1"
        sample_notebook.re_analyze_cell("cell1")
        sample_notebook.re_analyze_cell("cell2")

        executor = CellExecutor(sample_notebook)
        first = await executor.execute_cell("cell1", "x = 1")
        second = await executor.execute_cell("cell1", "x = 1")

        assert first.success is True
        assert first.execution_method == "executor"
        assert first.cache_hit is False
        assert first.remote_worker == "gpu-http-signed"
        assert first.remote_transport == "signed"
        assert first.remote_build_id is not None
        assert first.remote_build_state == "ready"
        assert first.remote_error_code is None
        assert first.outputs["x"]["preview"] == 1

        assert second.success is True
        assert second.cache_hit is True
        assert second.remote_build_id is None
        assert second.remote_build_state == "ready"
        assert second.remote_error_code is None

    def _signed_worker(self, session, executor_server, build_server):
        spec = {
            "url": executor_server["execute_url"],
            "transport": "signed",
            "strata_url": build_server["base_url"],
        }
        build_server["config"].transforms_config["notebook_workers"] = [
            {"name": "gpu-http-signed", "backend": "executor", "runtime_id": "a100", "config": spec}
        ]
        session.notebook_state.workers = [
            WorkerSpec(
                name="gpu-http-signed",
                backend=WorkerBackendType.EXECUTOR,
                runtime_id="a100",
                config=spec,
            )
        ]
        session.notebook_state.worker = "gpu-http-signed"
        cell1 = next(c for c in session.notebook_state.cells if c.id == "cell1")
        cell1.worker = "gpu-http-signed"
        cell1.source = "x = 1"
        session.re_analyze_cell("cell1")

    def _capture_manifest_metadata(self, monkeypatch):
        import strata.server as server_module

        signer = server_module._state.url_signer
        real = signer.generate_build_manifest
        seen: list[dict] = []

        def _capturing(*args, **kwargs):
            seen.append(kwargs["metadata"])
            return real(*args, **kwargs)

        monkeypatch.setattr(signer, "generate_build_manifest", _capturing)
        return seen

    @pytest.mark.asyncio
    async def test_the_signed_manifest_says_who_and_what_it_is_for(
        self,
        sample_notebook,
        notebook_executor_server,
        notebook_build_server,
        monkeypatch,
    ):
        """A dispatcher attributes the job and matches it to a cell from the manifest alone,
        without a GET /v1/builds round trip.
        """
        from strata.types import Principal

        self._signed_worker(sample_notebook, notebook_executor_server, notebook_build_server)
        seen = self._capture_manifest_metadata(monkeypatch)
        monkeypatch.setattr(
            "strata.auth.get_principal", lambda: Principal(id="alice", tenant="team-a")
        )

        result = await CellExecutor(sample_notebook).execute_cell("cell1", "x = 1")

        (metadata,) = seen
        assert metadata["principal"] == "alice"
        assert metadata["tenant"] == "team-a"
        assert metadata["notebook_id"] == sample_notebook.notebook_state.id
        assert metadata["cell_id"] == "cell1"
        cell = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        # The cell's own key (the one a rerun of the same computation shares), not
        # the transport hash the build row is keyed by.
        assert metadata["cell_provenance_hash"]
        assert metadata["cell_provenance_hash"] == cell.last_provenance_hash
        build = notebook_build_server["build_store"].get_build(result.remote_build_id)
        artifact = notebook_build_server["artifact_store"].get_artifact(
            build.artifact_id, build.version
        )
        assert metadata["cell_provenance_hash"] != artifact.provenance_hash

    @pytest.mark.asyncio
    async def test_the_env_reaches_the_worker_but_no_stored_row(
        self,
        sample_notebook,
        notebook_executor_server,
        notebook_build_server,
        monkeypatch,
    ):
        """Secrets ride only in the manifest; the artifact and build rows, which
        ``GET /v1/artifacts/{id}/v/{n}`` returns, name the keys and digest the values."""
        import json

        self._signed_worker(sample_notebook, notebook_executor_server, notebook_build_server)
        seen = self._capture_manifest_metadata(monkeypatch)
        cell1 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        source = 'import os\nprint(os.environ["MY_API_KEY"])\nx = 1'
        cell1.source = source
        cell1.env = {"MY_API_KEY": "sk-signed-secret"}
        sample_notebook.re_analyze_cell("cell1")

        result = await CellExecutor(sample_notebook).execute_cell("cell1", source)

        assert result.success, result.error
        assert "sk-signed-secret" in result.stdout
        (metadata,) = seen
        assert metadata["params"]["env"] == {"MY_API_KEY": "sk-signed-secret"}
        build = notebook_build_server["build_store"].get_build(result.remote_build_id)
        artifact = notebook_build_server["artifact_store"].get_artifact(
            build.artifact_id, build.version
        )
        assert build.params["env"]["names"] == ["MY_API_KEY"]
        assert "sk-signed-secret" not in json.dumps(build.params)
        assert json.loads(artifact.transform_spec)["params"]["env"]["names"] == ["MY_API_KEY"]
        assert "sk-signed-secret" not in artifact.transform_spec

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("cell_timeout", "floor", "expected"),
        [(3600.0, 600.0, 3600.0 + 600.0 + 300.0), (30.0, 7200.0, 7200.0)],
        ids=["outlives-a-long-cell", "setting-is-a-floor"],
    )
    async def test_signed_urls_outlive_provisioning_and_the_cell_timeout(
        self,
        sample_notebook,
        notebook_executor_server,
        notebook_build_server,
        monkeypatch,
        cell_timeout,
        floor,
        expected,
    ):
        """A cell that runs past signed_url_expiry_seconds must still upload and finalize."""
        import strata.server as server_module

        self._signed_worker(sample_notebook, notebook_executor_server, notebook_build_server)
        config = notebook_build_server["config"]
        monkeypatch.setattr(config, "signed_url_expiry_seconds", floor)
        monkeypatch.setattr(config, "worker_provisioning_timeout_seconds", 600.0)
        signer = server_module._state.url_signer
        real = signer.generate_build_manifest
        expiries: list[float] = []

        def _capturing(*args, **kwargs):
            expiries.append(kwargs["url_expiry_seconds"])
            return real(*args, **kwargs)

        monkeypatch.setattr(signer, "generate_build_manifest", _capturing)

        result = await CellExecutor(sample_notebook).execute_cell(
            "cell1", "x = 1", timeout_seconds=cell_timeout
        )

        assert result.success, result.error
        assert expiries == [expected]

    @pytest.mark.asyncio
    async def test_the_signed_manifest_is_built_off_the_event_loop(
        self,
        sample_notebook,
        notebook_executor_server,
        notebook_build_server,
        monkeypatch,
    ):
        """Presigning can call the cloud (a role, IAM signBlob), so it must not block the loop."""
        import asyncio

        import strata.server as server_module

        self._signed_worker(sample_notebook, notebook_executor_server, notebook_build_server)
        signer = server_module._state.url_signer
        real = signer.generate_build_manifest
        on_loop: list[bool] = []

        def _recording(*args, **kwargs):
            try:
                asyncio.get_running_loop()
                on_loop.append(True)
            except RuntimeError:
                on_loop.append(False)
            return real(*args, **kwargs)

        monkeypatch.setattr(signer, "generate_build_manifest", _recording)

        result = await CellExecutor(sample_notebook).execute_cell("cell1", "x = 1")

        assert result.success, result.error
        assert on_loop == [False]

    @pytest.mark.asyncio
    async def test_a_personal_server_says_what_but_not_who(
        self,
        sample_notebook,
        notebook_executor_server,
        notebook_build_server,
        monkeypatch,
    ):
        """No proxy, no principal: its jobs are attributed to the owner from the
        dispatch token. The notebook, cell and provenance are what only the
        manifest can say."""
        monkeypatch.setattr(notebook_build_server["config"], "deployment_mode", "personal")
        self._signed_worker(sample_notebook, notebook_executor_server, notebook_build_server)
        seen = self._capture_manifest_metadata(monkeypatch)

        result = await CellExecutor(sample_notebook).execute_cell("cell1", "x = 1")

        assert result.success, result.error
        (metadata,) = seen
        assert (metadata["principal"], metadata["tenant"]) == (None, None)
        assert metadata["notebook_id"] == sample_notebook.notebook_state.id
        assert metadata["cell_id"] == "cell1"
        assert metadata["cell_provenance_hash"]

    @pytest.mark.asyncio
    async def test_identity_does_not_change_what_is_cached(
        self,
        sample_notebook,
        notebook_executor_server,
        notebook_build_server,
        monkeypatch,
    ):
        """The fields ride in the manifest metadata, not in ``params``, which is
        hashed into the transport provenance: who ran a cell must not make the
        same computation a different cache entry."""
        from strata.types import Principal

        self._signed_worker(sample_notebook, notebook_executor_server, notebook_build_server)
        seen = self._capture_manifest_metadata(monkeypatch)
        executor = CellExecutor(sample_notebook)

        monkeypatch.setattr("strata.auth.get_principal", lambda: Principal(id="alice"))
        first = await executor.execute_cell_force("cell1", "x = 1")
        monkeypatch.setattr("strata.auth.get_principal", lambda: Principal(id="bob"))
        second = await executor.execute_cell_force("cell1", "x = 1")

        assert [m["principal"] for m in seen] == ["alice", "bob"]
        for field in ("principal", "tenant", "notebook_id", "cell_id", "cell_provenance_hash"):
            assert field not in seen[0]["params"]
        store = notebook_build_server["artifact_store"]
        builds = notebook_build_server["build_store"]
        hashes = [
            store.get_artifact(b.artifact_id, b.version).provenance_hash
            for b in (
                builds.get_build(first.remote_build_id),
                builds.get_build(second.remote_build_id),
            )
        ]
        assert hashes[0] == hashes[1]

    @pytest.mark.asyncio
    async def test_execute_supports_signed_http_executor_worker_with_class_instances(
        self,
        tmp_path,
        notebook_executor_server,
        notebook_build_server,
    ):
        """Signed executor workers should preserve exported class instances across cells."""
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        notebook_dir = create_notebook(tmp_path, "signed_http_class_instances")
        add_cell_to_notebook(notebook_dir, "cell1", None)
        add_cell_to_notebook(notebook_dir, "cell2", "cell1")
        add_cell_to_notebook(notebook_dir, "cell3", "cell2")

        write_cell(
            notebook_dir,
            "cell1",
            """
class Person:
    name = "John"
    age = 20

    def __str__(self):
        return f"{self.name}:{self.age}"
""".strip(),
        )
        write_cell(notebook_dir, "cell2", "p = Person()")
        write_cell(notebook_dir, "cell3", "rendered = str(p)")

        notebook_build_server["config"].transforms_config["notebook_workers"] = [
            {
                "name": "gpu-http-signed",
                "backend": "executor",
                "runtime_id": "gpu-http-signed-a100",
                "config": {
                    "url": notebook_executor_server["execute_url"],
                    "transport": "signed",
                    "strata_url": notebook_build_server["base_url"],
                },
            }
        ]

        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        session.notebook_state.workers = [
            WorkerSpec(
                name="gpu-http-signed",
                backend=WorkerBackendType.EXECUTOR,
                runtime_id="gpu-http-signed-a100",
                config={
                    "url": notebook_executor_server["execute_url"],
                    "transport": "signed",
                    "strata_url": notebook_build_server["base_url"],
                },
            )
        ]
        session.notebook_state.worker = "gpu-http-signed"
        for cell in session.notebook_state.cells:
            cell.worker = "gpu-http-signed"

        executor = CellExecutor(session)
        first = await executor.execute_cell("cell1", session.notebook_state.cells[0].source)
        second = await executor.execute_cell("cell2", session.notebook_state.cells[1].source)
        third = await executor.execute_cell("cell3", session.notebook_state.cells[2].source)

        assert first.success is True
        assert first.execution_method == "executor"
        assert first.remote_transport == "signed"
        assert first.remote_build_state == "ready"

        assert second.success is True
        assert second.execution_method == "executor"
        assert second.remote_transport == "signed"
        assert second.remote_build_state == "ready"
        assert second.outputs["p"]["content_type"] == "module/cell-instance"

        assert third.success is True
        assert third.execution_method == "executor"
        assert third.remote_transport == "signed"
        assert third.remote_build_state == "ready"
        assert third.outputs["rendered"]["preview"] == "John:20"

    @pytest.mark.asyncio
    async def test_execute_supports_signed_http_executor_worker_with_injection(
        self,
        sample_notebook,
        notebook_executor_server,
        notebook_build_server,
    ):
        """A shared def that hydrates a runtime value works over the signed
        transport: the injected blob is staged as a signed-URL artifact and the
        worker fetches it to hydrate the synthetic module."""
        worker_cfg = {
            "url": notebook_executor_server["execute_url"],
            "transport": "signed",
            "strata_url": notebook_build_server["base_url"],
        }
        notebook_build_server["config"].transforms_config["notebook_workers"] = [
            {
                "name": "gpu-http-signed",
                "backend": "executor",
                "runtime_id": "gpu-http-signed-a100",
                "config": worker_cfg,
            }
        ]
        sample_notebook.notebook_state.workers = [
            WorkerSpec(
                name="gpu-http-signed",
                backend=WorkerBackendType.EXECUTOR,
                runtime_id="gpu-http-signed-a100",
                config=worker_cfg,
            )
        ]
        sample_notebook.notebook_state.worker = "gpu-http-signed"
        cell1 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell2 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell2")
        cell1.worker = "gpu-http-signed"
        cell2.worker = "gpu-http-signed"
        cell1.source = "base = 10 + len('abc')\n\ndef addbase(y):\n    return base + y"
        cell2.source = "result = addbase(5)"
        sample_notebook.re_analyze_cell("cell1")
        sample_notebook.re_analyze_cell("cell2")

        executor = CellExecutor(sample_notebook)
        first = await executor.execute_cell("cell1", cell1.source)
        assert first.success is True
        assert first.remote_transport == "signed"
        assert first.outputs["addbase"]["content_type"] == "module/cell"

        second = await executor.execute_cell("cell2", cell2.source)
        assert second.success is True
        assert second.remote_transport == "signed"
        assert second.outputs["result"]["preview"] == 18

    @pytest.mark.asyncio
    async def test_execute_supports_signed_http_executor_worker_in_personal_mode(
        self,
        sample_notebook,
        notebook_executor_server,
        notebook_personal_server,
    ):
        """Signed notebook transport should work in personal mode without server transforms."""
        sample_notebook.notebook_state.workers = [
            WorkerSpec(
                name="gpu-http-signed",
                backend=WorkerBackendType.EXECUTOR,
                runtime_id="gpu-http-signed-a100",
                config={
                    "url": notebook_executor_server["execute_url"],
                    "transport": "signed",
                    "strata_url": notebook_personal_server["base_url"],
                },
            )
        ]
        sample_notebook.notebook_state.worker = "gpu-http-signed"
        cell1 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell1.worker = "gpu-http-signed"
        cell1.source = "x = 1"
        sample_notebook.re_analyze_cell("cell1")

        result = await CellExecutor(sample_notebook).execute_cell("cell1", "x = 1")

        assert result.success is True
        assert result.execution_method == "executor"
        assert result.remote_worker == "gpu-http-signed"
        assert result.remote_transport == "signed"
        assert result.remote_build_id is not None
        assert result.remote_build_state == "ready"
        assert result.remote_error_code is None
        assert result.outputs["x"]["preview"] == 1

    @pytest.mark.asyncio
    async def test_execute_signed_http_executor_marks_build_failed_on_transport_error(
        self,
        sample_notebook,
        notebook_executor_server,
        notebook_build_server,
    ):
        """Signed transport failures should leave no pending/building notebook builds."""
        notebook_build_server["config"].transforms_config["notebook_workers"] = [
            {
                "name": "gpu-http-signed",
                "backend": "executor",
                "runtime_id": "gpu-http-signed-a100",
                "config": {
                    "url": notebook_executor_server["execute_url"],
                    "transport": "signed",
                    "strata_url": "http://127.0.0.1:9",
                },
            }
        ]
        sample_notebook.notebook_state.worker = "gpu-http-signed"
        cell = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell.worker = "gpu-http-signed"

        result = await CellExecutor(sample_notebook).execute_cell("cell1", "x = 1")

        assert result.success is False
        assert "Remote executor 'gpu-http-signed' returned 502:" in str(result.error)
        assert "Notebook bundle transfer failed:" in str(result.error)
        assert result.remote_worker == "gpu-http-signed"
        assert result.remote_transport == "signed"
        assert result.remote_build_id is not None
        assert result.remote_build_state == "failed"
        assert result.remote_error_code == "EXECUTOR_HTTP_ERROR"

        stats = notebook_build_server["build_store"].get_stats()
        assert stats["failed"] == 1
        assert stats["pending"] == 0
        assert stats["building"] == 0

    @pytest.mark.asyncio
    async def test_cancelled_signed_http_executor_marks_build_failed(
        self,
        sample_notebook,
        notebook_executor_server,
        notebook_build_server,
        monkeypatch,
    ):
        """Cancelling signed transport execution should fail the in-flight build."""
        started = threading.Event()

        async def _slow_run_harness(
            harness_path: Path,
            manifest_path: Path,
            timeout_seconds: float,
            *,
            in_flight: dict | None = None,
            build_id: str | None = None,
            log_url: str | None = None,
            env: dict[str, str] | None = None,
            interpreter: Path | None = None,
        ) -> dict[str, object]:
            del (
                harness_path,
                manifest_path,
                timeout_seconds,
                in_flight,
                build_id,
                log_url,
                env,
                interpreter,
            )
            started.set()
            await asyncio.sleep(0.5)
            return {
                "success": True,
                "variables": {
                    "x": {
                        "content_type": "json/object",
                        "file": "x.json",
                        "preview": 1,
                    }
                },
                "stdout": "",
                "stderr": "",
                "mutation_warnings": [],
            }

        monkeypatch.setattr(
            "strata.notebook.remote_executor._run_harness",
            _slow_run_harness,
        )

        notebook_build_server["config"].transforms_config["notebook_workers"] = [
            {
                "name": "gpu-http-signed",
                "backend": "executor",
                "runtime_id": "gpu-http-signed-a100",
                "config": {
                    "url": notebook_executor_server["execute_url"],
                    "transport": "signed",
                    "strata_url": notebook_build_server["base_url"],
                },
            }
        ]
        sample_notebook.notebook_state.worker = "gpu-http-signed"
        cell = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell.worker = "gpu-http-signed"

        task = asyncio.create_task(CellExecutor(sample_notebook).execute_cell("cell1", "x = 1"))
        assert await asyncio.to_thread(started.wait, 2.0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        stats = notebook_build_server["build_store"].get_stats()
        assert stats["failed"] == 1
        assert stats["pending"] == 0
        assert stats["building"] == 0

    @pytest.mark.asyncio
    async def test_execute_rejects_file_mounts_for_http_executor_worker(
        self,
        sample_notebook,
        notebook_executor_server,
        tmp_path: Path,
    ):
        """HTTP executor workers should reject notebook-declared file mounts."""
        sample_notebook.notebook_state.workers = [
            WorkerSpec(
                name="gpu-http",
                backend=WorkerBackendType.EXECUTOR,
                runtime_id="gpu-http-a100",
                config={"url": notebook_executor_server["execute_url"]},
            )
        ]
        cell = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell.worker = "gpu-http"
        cell.mounts = [
            MountSpec(
                name="raw_data",
                uri=f"file://{tmp_path}",
                mode=MountMode.READ_ONLY,
            )
        ]

        result = await CellExecutor(sample_notebook).execute_cell("cell1", "x = 1")

        assert result.success is False
        assert result.error == (
            "Execution failed: Remote executor workers do not support file:// mounts: 'raw_data'"
        )

    @pytest.mark.asyncio
    async def test_execution_duration(self, sample_notebook):
        executor = CellExecutor(sample_notebook)

        source = "x = 42"
        result = await executor.execute_cell("cell1", source)

        assert result.success is True
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_execute_function_definition(self, sample_notebook):
        executor = CellExecutor(sample_notebook)

        source = """
def add(a, b):
    return a + b

result = add(2, 3)
"""
        result = await executor.execute_cell("cell1", source)

        assert result.success is True
        # Function definition is not captured, but result is
        assert "add" in result.outputs
        assert "result" in result.outputs
        assert result.outputs["result"]["preview"] == 5

    @pytest.mark.asyncio
    async def test_execute_supports_cross_cell_function_definition(self, sample_notebook):
        """Top-level functions should be reusable across cells via module export."""
        cell1 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell2 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell2")
        cell1.source = """
import math

def area(r):
    return math.pi * r * r
"""
        cell2.source = "result = round(area(2), 5)"
        sample_notebook.re_analyze_cell("cell1")
        sample_notebook.re_analyze_cell("cell2")

        executor = CellExecutor(sample_notebook)
        first = await executor.execute_cell("cell1", cell1.source)
        second = await executor.execute_cell("cell2", cell2.source)

        assert first.success is True
        assert first.outputs["area"]["content_type"] == "module/cell"
        assert second.success is True
        assert second.outputs["result"]["preview"] == 12.56637

    @pytest.mark.asyncio
    async def test_execute_supports_cross_cell_class_definition(self, sample_notebook):
        """Top-level classes should be reusable across cells via module export."""
        cell1 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell2 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell2")
        cell1.source = """
class Box:
    def __init__(self, x):
        self.x = x
"""
        cell2.source = "value = Box(3).x"
        sample_notebook.re_analyze_cell("cell1")
        sample_notebook.re_analyze_cell("cell2")

        executor = CellExecutor(sample_notebook)
        first = await executor.execute_cell("cell1", cell1.source)
        second = await executor.execute_cell("cell2", cell2.source)

        assert first.success is True
        assert first.outputs["Box"]["content_type"] == "module/cell"
        assert second.success is True
        assert second.outputs["value"]["preview"] == 3

    @pytest.mark.asyncio
    async def test_execute_supports_exported_class_instances_across_cells(self, tmp_path):
        """Instances of exported notebook classes should round-trip to downstream cells."""
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        notebook_dir = create_notebook(tmp_path, "class_instances")
        add_cell_to_notebook(notebook_dir, "cell1", None)
        add_cell_to_notebook(notebook_dir, "cell2", "cell1")
        add_cell_to_notebook(notebook_dir, "cell3", "cell2")

        write_cell(
            notebook_dir,
            "cell1",
            """
class Person:
    name = "John"
    age = 20

    def __str__(self):
        return f"{self.name}:{self.age}"
""".strip(),
        )
        write_cell(notebook_dir, "cell2", "p = Person()")
        write_cell(notebook_dir, "cell3", "rendered = str(p)")

        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        session.refresh_environment_runtime()
        executor = CellExecutor(session)

        first = await executor.execute_cell("cell1", session.notebook_state.cells[0].source)
        second = await executor.execute_cell("cell2", session.notebook_state.cells[1].source)
        third = await executor.execute_cell("cell3", session.notebook_state.cells[2].source)

        assert first.success is True
        assert first.outputs["Person"]["content_type"] == "module/cell"
        assert second.success is True
        assert second.outputs["p"]["content_type"] == "module/cell-instance"
        assert third.success is True
        assert third.outputs["rendered"]["preview"] == "John:20"

    @pytest.mark.asyncio
    async def test_execute_supports_slot_based_class_instances_across_cells(self, tmp_path):
        """Slot-based exported instances should round-trip across notebook cells."""
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        notebook_dir = create_notebook(tmp_path, "slot_instances")
        add_cell_to_notebook(notebook_dir, "cell1", None)
        add_cell_to_notebook(notebook_dir, "cell2", "cell1")
        add_cell_to_notebook(notebook_dir, "cell3", "cell2")

        write_cell(
            notebook_dir,
            "cell1",
            """
class Person:
    __slots__ = ("name", "age")

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"{self.name}:{self.age}"
""".strip(),
        )
        write_cell(notebook_dir, "cell2", 'p = Person("Ada", 10)')
        write_cell(notebook_dir, "cell3", "rendered = str(p)")

        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        session.refresh_environment_runtime()
        executor = CellExecutor(session)

        first = await executor.execute_cell("cell1", session.notebook_state.cells[0].source)
        second = await executor.execute_cell("cell2", session.notebook_state.cells[1].source)
        third = await executor.execute_cell("cell3", session.notebook_state.cells[2].source)

        assert first.success is True
        assert second.success is True
        assert second.outputs["p"]["content_type"] == "module/cell-instance"
        assert third.success is True
        assert third.outputs["rendered"]["preview"] == "Ada:10"

    @pytest.mark.asyncio
    async def test_execute_shares_export_with_same_cell_runtime_state(self, sample_notebook):
        """A def closing over a same-cell runtime value is shareable: the value is stored and
        hydrated into the synthetic module.
        """
        cell1 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell2 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell2")
        # ``x = len([])`` is a non-literal runtime assignment ``add`` closes over.
        cell1.source = """
x = len([])

def add(y):
    return x + y
"""
        cell2.source = "result = add(2)"
        sample_notebook.re_analyze_cell("cell1")
        sample_notebook.re_analyze_cell("cell2")

        executor = CellExecutor(sample_notebook)
        result1 = await executor.execute_cell("cell1", cell1.source)
        assert result1.success is True
        assert result1.outputs["add"]["content_type"] == "module/cell"

        result2 = await executor.execute_cell("cell2", cell2.source)
        assert result2.success is True
        # x = len([]) = 0, so add(2) = 2.
        assert result2.outputs["result"]["preview"] == 2

    @pytest.mark.asyncio
    async def test_execute_rejects_cross_cell_export_with_top_level_lambda(self, sample_notebook):
        """Top-level lambdas should fail with a targeted exportability error."""
        cell1 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell2 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell2")
        cell1.source = "add = lambda y: y + 1"
        cell2.source = "result = add(2)"
        sample_notebook.re_analyze_cell("cell1")
        sample_notebook.re_analyze_cell("cell2")

        result = await CellExecutor(sample_notebook).execute_cell("cell1", cell1.source)

        assert result.success is False
        assert result.error is not None
        assert "cannot be shared across cells yet" in result.error
        assert "top-level lambdas are not shareable across cells" in result.error

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.warm_pool
    async def test_execute_uses_warm_pool_when_available(self, sample_notebook):
        sample_notebook.ensure_venv_synced()
        pool = WarmProcessPool(
            sample_notebook.path,
            pool_size=1,
            python_executable=sample_notebook.venv_python or Path("python"),
        )
        await pool.start()

        try:
            executor = CellExecutor(sample_notebook, pool)
            result = await executor.execute_cell("cell1", "x = 1 + 1")

            assert result.success is True
            assert result.execution_method == "warm"
            assert result.outputs["x"]["preview"] == 2
        finally:
            await pool.drain()

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.warm_pool
    async def test_warm_execution_applies_env_annotations(self, sample_notebook):
        """Warm workers should honor @env overrides the same way cold execution does."""
        sample_notebook.ensure_venv_synced()
        pool = WarmProcessPool(
            sample_notebook.path,
            pool_size=1,
            python_executable=sample_notebook.venv_python or Path("python"),
        )
        await pool.start()

        try:
            executor = CellExecutor(sample_notebook, pool)
            result = await executor.execute_cell(
                "cell1",
                """
# @env NOTEBOOK_TOKEN=warm-secret
import os
token = os.getenv("NOTEBOOK_TOKEN")
""",
            )

            assert result.success is True
            assert result.execution_method == "warm"
            assert result.outputs["token"]["preview"] == "warm-secret"
        finally:
            await pool.drain()

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.warm_pool
    async def test_warm_execution_injects_mount_paths(self, sample_notebook, tmp_path: Path):
        """Warm workers should inject prepared mount paths the same way cold execution does."""
        mount_dir = tmp_path / "mounted-data"
        mount_dir.mkdir()
        (mount_dir / "data.txt").write_text("hello mount", encoding="utf-8")

        cell = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell.mounts = [
            MountSpec(
                name="raw_data",
                uri=f"file://{mount_dir}",
                mode=MountMode.READ_ONLY,
            )
        ]

        sample_notebook.ensure_venv_synced()
        pool = WarmProcessPool(
            sample_notebook.path,
            pool_size=1,
            python_executable=sample_notebook.venv_python or Path("python"),
        )
        await pool.start()

        try:
            executor = CellExecutor(sample_notebook, pool)
            result = await executor.execute_cell(
                "cell1",
                'text = (raw_data / "data.txt").read_text()',
            )

            assert result.success is True
            assert result.execution_method == "warm"
            assert result.outputs["text"]["preview"] == "hello mount"
        finally:
            await pool.drain()

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.warm_pool
    async def test_warm_execution_supports_cross_cell_function_exports(self, sample_notebook):
        """Warm workers should receive synthetic module exports as upstream inputs."""
        cell1 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell2 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell2")
        cell1.source = """
def add(a, b):
    return a + b
"""
        cell2.source = "result = add(2, 3)"
        sample_notebook.re_analyze_cell("cell1")
        sample_notebook.re_analyze_cell("cell2")

        cold_executor = CellExecutor(sample_notebook)
        first = await cold_executor.execute_cell("cell1", cell1.source)
        assert first.success is True
        assert first.outputs["add"]["content_type"] == "module/cell"

        sample_notebook.ensure_venv_synced()
        pool = WarmProcessPool(
            sample_notebook.path,
            pool_size=1,
            python_executable=sample_notebook.venv_python or Path("python"),
        )
        await pool.start()

        try:
            executor = CellExecutor(sample_notebook, pool)
            second = await executor.execute_cell("cell2", cell2.source)

            assert second.success is True
            assert second.execution_method == "warm"
            assert second.outputs["result"]["preview"] == 5
        finally:
            await pool.drain()

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.warm_pool
    async def test_warm_execution_supports_exported_class_instances(self, tmp_path):
        """Warm workers should deserialize instances of exported notebook classes."""
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        notebook_dir = create_notebook(tmp_path, "warm_class_instances")
        add_cell_to_notebook(notebook_dir, "cell1", None)
        add_cell_to_notebook(notebook_dir, "cell2", "cell1")
        add_cell_to_notebook(notebook_dir, "cell3", "cell2")

        write_cell(
            notebook_dir,
            "cell1",
            """
class Person:
    name = "John"
    age = 20

    def __str__(self):
        return f"{self.name}:{self.age}"
""".strip(),
        )
        write_cell(notebook_dir, "cell2", "p = Person()")
        write_cell(notebook_dir, "cell3", "rendered = str(p)")

        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        session.refresh_environment_runtime()
        cold_executor = CellExecutor(session)

        first = await cold_executor.execute_cell("cell1", session.notebook_state.cells[0].source)
        second = await cold_executor.execute_cell("cell2", session.notebook_state.cells[1].source)
        assert first.success is True
        assert second.success is True
        assert second.outputs["p"]["content_type"] == "module/cell-instance"

        session.ensure_venv_synced()
        pool = WarmProcessPool(
            session.path,
            pool_size=1,
            python_executable=session.venv_python or Path("python"),
        )
        await pool.start()

        try:
            executor = CellExecutor(session, pool)
            third = await executor.execute_cell("cell3", session.notebook_state.cells[2].source)

            assert third.success is True
            assert third.execution_method == "warm"
            assert third.outputs["rendered"]["preview"] == "John:20"
        finally:
            await pool.drain()

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.warm_pool
    async def test_warm_execution_reports_same_mutation_warnings_as_cold(self, sample_notebook):
        """Warm workers should preserve mutation warnings emitted by cold execution."""
        cell1 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell1")
        cell2 = next(c for c in sample_notebook.notebook_state.cells if c.id == "cell2")
        cell1.source = "x = 1"
        cell2.source = "del x\ny = 2"
        sample_notebook.re_analyze_cell("cell1")
        sample_notebook.re_analyze_cell("cell2")

        cold_result = await CellExecutor(sample_notebook).execute_cell("cell2", cell2.source)
        assert cold_result.success is True
        assert cold_result.mutation_warnings
        assert cold_result.mutation_warnings[0]["var_name"] == "x"

        sample_notebook.ensure_venv_synced()
        pool = WarmProcessPool(
            sample_notebook.path,
            pool_size=1,
            python_executable=sample_notebook.venv_python or Path("python"),
        )
        await pool.start()

        try:
            # The cold run stored cell2's (empty) console, so a plain run would be a cache hit;
            # rerun bypasses it so the warm path actually executes.
            warm_result = await CellExecutor(sample_notebook, pool).execute_cell_rerun(
                "cell2", cell2.source
            )
            assert warm_result.success is True
            assert warm_result.execution_method == "warm"
            assert warm_result.mutation_warnings == cold_result.mutation_warnings
        finally:
            await pool.drain()


class TestPromptCellExecution:
    """Tests for prompt cell execution via LLM."""

    @pytest.mark.asyncio
    async def test_prompt_cell_returns_error_when_llm_not_configured(self, tmp_path):
        """Prompt cells should return a clear error when no LLM provider is set."""
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        notebook_dir = create_notebook(tmp_path, "prompt_test")
        add_cell_to_notebook(notebook_dir, "p1", language="prompt")
        write_cell(notebook_dir, "p1", "What is Python?")

        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        cell = next(c for c in session.notebook_state.cells if c.id == "p1")
        assert cell.language == "prompt"

        executor = CellExecutor(session)
        result = await executor.execute_cell("p1", cell.source)

        assert result.success is False
        assert result.cell_id == "p1"
        assert "not configured" in (result.error or "").lower()
        # CLI and agent users have no panel; the error names the file they can edit.
        assert "[env] in notebook.toml" in (result.error or "")
        assert result.execution_method == "llm"

    @pytest.mark.asyncio
    async def test_prompt_cell_calls_llm_and_stores_artifact(self, tmp_path):
        """Prompt cells should call the LLM and store the result as an artifact."""
        from unittest.mock import AsyncMock, patch

        from strata.notebook.llm import LlmCompletionResult
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        notebook_dir = create_notebook(tmp_path, "prompt_exec_test")
        add_cell_to_notebook(notebook_dir, "c1")
        write_cell(notebook_dir, "c1", "x = 42")
        add_cell_to_notebook(notebook_dir, "p1", after_cell_id="c1", language="prompt")
        write_cell(notebook_dir, "p1", "# @name answer\nWhat is {{ x }}?")

        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        session.refresh_environment_runtime()
        session.notebook_state.env["STRATA_AI_API_KEY"] = "sk-test"

        # Execute c1 first so x has an artifact
        executor = CellExecutor(session)
        r1 = await executor.execute_cell("c1", "x = 42")
        assert r1.success

        mock_result = LlmCompletionResult(
            content="42 is the answer to everything.",
            model="test-model",
            input_tokens=10,
            output_tokens=8,
        )

        with patch(
            "strata.notebook.prompt_executor.chat_completion",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await executor.execute_cell("p1", "# @name answer\nWhat is {{ x }}?")

        assert result.success is True
        assert result.cell_id == "p1"
        assert "answer" in result.outputs
        assert "42 is the answer" in str(result.outputs["answer"].get("preview", ""))
        assert result.execution_method == "llm"
        assert result.artifact_uri is not None

    @pytest.mark.asyncio
    async def test_prompt_cell_cache_hit_reuses_artifact(self, tmp_path):
        """Prompt cells should hit cache on identical reruns."""
        from unittest.mock import AsyncMock, patch

        from strata.notebook.llm import LlmCompletionResult
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        notebook_dir = create_notebook(tmp_path, "prompt_cache_test")
        add_cell_to_notebook(notebook_dir, "c1")
        write_cell(notebook_dir, "c1", "x = 42")
        add_cell_to_notebook(notebook_dir, "p1", after_cell_id="c1", language="prompt")
        write_cell(notebook_dir, "p1", "# @name answer\nWhat is {{ x }}?")

        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        session.refresh_environment_runtime()
        session.notebook_state.env["STRATA_AI_API_KEY"] = "sk-test"
        executor = CellExecutor(session)
        assert (await executor.execute_cell("c1", "x = 42")).success

        mock_result = LlmCompletionResult(
            content="42 is the answer to everything.",
            model="test-model",
            input_tokens=10,
            output_tokens=8,
        )

        with patch(
            "strata.notebook.prompt_executor.chat_completion",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_chat:
            first = await executor.execute_cell("p1", "# @name answer\nWhat is {{ x }}?")
            second = await executor.execute_cell("p1", "# @name answer\nWhat is {{ x }}?")

        assert first.success is True
        assert first.cache_hit is False
        assert second.success is True
        assert second.cache_hit is True
        assert second.artifact_uri is not None
        assert mock_chat.await_count == 1

    @pytest.mark.asyncio
    async def test_prompt_cell_passes_temperature_to_llm(self, tmp_path):
        """Prompt-cell @temperature should be sent to the provider call."""
        import unittest.mock as mock

        from strata.notebook.llm import LlmCompletionResult
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        notebook_dir = create_notebook(tmp_path, "prompt_temp_test")
        add_cell_to_notebook(notebook_dir, "p1", language="prompt")
        write_cell(notebook_dir, "p1", "# @temperature 0.7\nHello")

        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        session.notebook_state.env["STRATA_AI_API_KEY"] = "sk-test"
        executor = CellExecutor(session)
        captured: dict[str, object] = {}

        async def _fake_chat_completion(
            config,
            messages,
            *,
            temperature=None,
            output_type=None,
            output_schema=None,
        ):
            captured["config"] = config
            captured["messages"] = messages
            captured["temperature"] = temperature
            captured["output_type"] = output_type
            captured["output_schema"] = output_schema
            return LlmCompletionResult(
                content="hello",
                model="test-model",
                input_tokens=3,
                output_tokens=2,
            )

        with mock.patch("strata.notebook.prompt_executor.chat_completion", _fake_chat_completion):
            result = await executor.execute_cell("p1", "# @temperature 0.7\nHello")

        assert result.success is True
        assert captured["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_prompt_cell_dag_connects_to_upstream(self, tmp_path):
        """Prompt cell {{ var }} references create correct DAG edges."""
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        notebook_dir = create_notebook(tmp_path, "prompt_dag_test")
        add_cell_to_notebook(notebook_dir, "c1")
        write_cell(notebook_dir, "c1", "data = [1, 2, 3]")
        add_cell_to_notebook(notebook_dir, "p1", after_cell_id="c1", language="prompt")
        write_cell(notebook_dir, "p1", "# @name summary\nSummarize {{ data }}")

        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)

        p1 = next(c for c in session.notebook_state.cells if c.id == "p1")
        assert p1.language == "prompt"
        assert "data" in p1.references
        assert p1.defines == ["summary"]
        assert "c1" in p1.upstream_ids

        c1 = next(c for c in session.notebook_state.cells if c.id == "c1")
        assert "p1" in c1.downstream_ids


class TestLoopCellExecution:
    """End-to-end loop cell execution through the real harness subprocess.

    Covers carry resolution, per-iteration spawns, ``@loop_until`` termination,
    ``start_from`` forking and the per-iteration artifact ids.
    """

    @pytest.fixture
    def loop_notebook(self, tmp_path):
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        notebook_dir = create_notebook(tmp_path, "loop_test")
        # Upstream cell seeds the carry.
        add_cell_to_notebook(notebook_dir, "seed")
        write_cell(notebook_dir, "seed", "state = {'n': 0, 'history': []}")
        # The loop cell must carry `state` and rebind it each iteration.
        add_cell_to_notebook(notebook_dir, "loop", after_cell_id="seed")

        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        session.refresh_environment_runtime()
        return notebook_dir, session

    @pytest.mark.asyncio
    async def test_loop_runs_until_max_iter(self, loop_notebook):
        """Without ``@loop_until`` the loop runs exactly ``max_iter`` times."""
        from strata.notebook.writer import write_cell

        notebook_dir, session = loop_notebook
        loop_source = (
            "# @loop max_iter=3 carry=state\n"
            "state = {'n': state['n'] + 1, 'history': state['history'] + [state['n']]}\n"
        )
        write_cell(notebook_dir, "loop", loop_source)
        session.reload()

        executor = CellExecutor(session)

        await executor.execute_cell("seed", "state = {'n': 0, 'history': []}")
        result = await executor.execute_cell("loop", loop_source)

        assert result.success, result.error
        assert result.execution_method == "loop"
        # The result's top-level artifact_uri is the canonical (non-iter)
        # id so downstream cells can read it via the normal DAG path; the
        # per-iteration artifacts live under ``@iter={k}`` suffixes.
        assert result.artifact_uri is not None
        assert "@iter=" not in result.artifact_uri
        # Final iteration is k=2 with max_iter=3.
        artifact_mgr = session.get_artifact_manager()
        assert artifact_mgr.get_iteration_artifact("loop", "state", 2) is not None

    @pytest.mark.asyncio
    async def test_loop_until_terminates_early(self, loop_notebook):
        """``@loop_until`` terminates as soon as the predicate is truthy."""
        from strata.notebook.writer import write_cell

        notebook_dir, session = loop_notebook
        loop_source = (
            "# @loop max_iter=10 carry=state\n"
            "# @loop_until state['n'] >= 3\n"
            "state = {'n': state['n'] + 1, 'history': state['history'] + [state['n']]}\n"
        )
        write_cell(notebook_dir, "loop", loop_source)
        session.reload()

        executor = CellExecutor(session)
        await executor.execute_cell("seed", "state = {'n': 0, 'history': []}")
        result = await executor.execute_cell("loop", loop_source)

        assert result.success, result.error
        # The top-level artifact_uri is the canonical (non-iter) URI of
        # the final iteration's carry; per-iter URIs are separate.
        assert result.artifact_uri is not None
        assert "@iter=" not in result.artifact_uri

        # All three iteration artifacts should exist; the loop terminated
        # at iter=2 because state['n'] becomes 3 on iter 2.
        artifact_mgr = session.get_artifact_manager()
        for k in range(3):
            iter_artifact = artifact_mgr.get_iteration_artifact("loop", "state", k)
            assert iter_artifact is not None, f"iter={k} artifact missing"

    @pytest.mark.asyncio
    async def test_loop_carry_missing_from_outputs_is_surfaced(self, loop_notebook):
        """If the body forgets to rebind the carry, the cell fails with a
        clear error and the loop does not silently reuse the old value."""
        from strata.notebook.writer import write_cell

        notebook_dir, session = loop_notebook
        # The body mutates in place but never rebinds `state`, so the harness emits no
        # fresh ``state`` output; the loop must surface this.
        loop_source = "# @loop max_iter=2 carry=state\nstate['n'] += 1\n"
        write_cell(notebook_dir, "loop", loop_source)
        session.reload()

        executor = CellExecutor(session)
        await executor.execute_cell("seed", "state = {'n': 0, 'history': []}")
        result = await executor.execute_cell("loop", loop_source)

        assert not result.success
        assert "carry" in (result.error or "").lower()
        assert result.execution_method == "loop"

    @pytest.mark.asyncio
    async def test_loop_body_failure_returns_error(self, loop_notebook):
        """A Python error inside the cell body stops the loop cleanly."""
        from strata.notebook.writer import write_cell

        notebook_dir, session = loop_notebook
        loop_source = (
            "# @loop max_iter=5 carry=state\n"
            "raise RuntimeError('boom at iter ' + str(state['n']))\n"
        )
        write_cell(notebook_dir, "loop", loop_source)
        session.reload()

        executor = CellExecutor(session)
        await executor.execute_cell("seed", "state = {'n': 0, 'history': []}")
        result = await executor.execute_cell("loop", loop_source)

        assert not result.success
        assert "boom" in (result.error or "")

    @pytest.mark.asyncio
    async def test_loop_start_from_seeds_from_prior_iteration(self, tmp_path):
        """Forking a loop cell: duplicate cell with ``start_from`` seeds
        iter 0 from the donor cell's persisted iteration artifact."""
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        notebook_dir = create_notebook(tmp_path, "loop_fork_test")
        add_cell_to_notebook(notebook_dir, "seed")
        write_cell(notebook_dir, "seed", "state = {'n': 0}")
        add_cell_to_notebook(notebook_dir, "donor", after_cell_id="seed")
        write_cell(
            notebook_dir,
            "donor",
            "# @loop max_iter=3 carry=state\nstate = {'n': state['n'] + 1}\n",
        )
        add_cell_to_notebook(notebook_dir, "forked", after_cell_id="donor")
        write_cell(
            notebook_dir,
            "forked",
            (
                "# @loop max_iter=2 carry=state start_from=donor@iter=1\n"
                "state = {'n': state['n'] * 10}\n"
            ),
        )

        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        session.refresh_environment_runtime()
        executor = CellExecutor(session)

        await executor.execute_cell("seed", "state = {'n': 0}")
        donor_src = "# @loop max_iter=3 carry=state\nstate = {'n': state['n'] + 1}\n"
        donor_result = await executor.execute_cell("donor", donor_src)
        assert donor_result.success, donor_result.error

        # donor iter 1 should hold state['n'] == 2 (seed 0 → iter0 1 → iter1 2).
        artifact_mgr = session.get_artifact_manager()
        donor_iter1_blob = artifact_mgr.load_iteration_blob("donor", "state", 1)
        assert donor_iter1_blob is not None

        forked_src = (
            "# @loop max_iter=2 carry=state start_from=donor@iter=1\n"
            "state = {'n': state['n'] * 10}\n"
        )
        forked_result = await executor.execute_cell("forked", forked_src)
        assert forked_result.success, forked_result.error

        # Forked iter 0 multiplies the donor's iter 1 state (n=2) → 20.
        forked_iter0_blob = artifact_mgr.load_iteration_blob("forked", "state", 0)
        assert forked_iter0_blob is not None
        import json

        forked_iter0 = json.loads(forked_iter0_blob)
        assert forked_iter0 == {"n": 20}

    def test_a_start_from_seed_overtaken_by_a_duplicate_reads_its_canonical(self, loop_notebook):
        """A seed version dedup overtook has no bytes of its own; the seed reads through it."""
        from strata.artifact_store import TransformSpec
        from strata.notebook.annotations import LoopAnnotation

        _, session = loop_notebook
        artifact_mgr = session.get_artifact_manager()
        store = artifact_mgr.artifact_store
        spec = TransformSpec(executor="notebook/cell@v1", params={}, inputs=[])
        for artifact_id in ("canonical", artifact_mgr.cell_artifact_id("donor", "state", 1)):
            version = store.create_artifact(artifact_id, provenance_hash="p", transform_spec=spec)
            store.write_blob(artifact_id, version, b"seed-bytes")
            store.finalize_artifact(artifact_id, version, "", 0, 10)

        loop = LoopAnnotation(max_iter=2, carry="state", start_from_cell="donor", start_from_iter=1)
        blob, _ = CellExecutor(session)._resolve_loop_seed("loop", loop)
        assert blob == b"seed-bytes"

    @pytest.mark.asyncio
    async def test_loop_iteration_progress_callback_fires_per_iter(self, loop_notebook):
        """``on_iteration_complete`` fires once per completed iteration with that iteration's
        artifact URI; the WS handler broadcasts it as ``cell_iteration_progress``.
        """
        from strata.notebook.writer import write_cell

        notebook_dir, session = loop_notebook
        loop_source = (
            "# @loop max_iter=3 carry=state\n"
            "state = {'n': state['n'] + 1, 'history': state['history'] + [state['n']]}\n"
        )
        write_cell(notebook_dir, "loop", loop_source)
        session.reload()

        executor = CellExecutor(session)
        progress_events: list[dict] = []

        async def _record(event: dict) -> None:
            progress_events.append(event)

        executor.on_iteration_complete = _record

        await executor.execute_cell("seed", "state = {'n': 0, 'history': []}")
        result = await executor.execute_cell("loop", loop_source)

        assert result.success, result.error
        assert [event["iteration"] for event in progress_events] == [0, 1, 2]
        assert all(event["max_iter"] == 3 for event in progress_events)
        assert all(event["cell_id"] == "loop" for event in progress_events)
        assert all(
            event["artifact_uri"].endswith(f"@iter={event['iteration']}@v=1")
            for event in progress_events
        )
        assert all("duration_ms" in event for event in progress_events)

    @pytest.mark.asyncio
    async def test_loop_progress_stops_when_until_reached(self, loop_notebook):
        """Early termination via ``@loop_until`` is visible on the last
        progress event so the UI can mark the loop as completed."""
        from strata.notebook.writer import write_cell

        notebook_dir, session = loop_notebook
        loop_source = (
            "# @loop max_iter=10 carry=state\n"
            "# @loop_until state['n'] >= 2\n"
            "state = {'n': state['n'] + 1, 'history': state['history'] + [state['n']]}\n"
        )
        write_cell(notebook_dir, "loop", loop_source)
        session.reload()

        executor = CellExecutor(session)
        progress_events: list[dict] = []

        async def _record(event: dict) -> None:
            progress_events.append(event)

        executor.on_iteration_complete = _record

        await executor.execute_cell("seed", "state = {'n': 0, 'history': []}")
        await executor.execute_cell("loop", loop_source)

        assert [event["iteration"] for event in progress_events] == [0, 1]
        assert progress_events[-1]["until_reached"] is True
        # Earlier iterations must not mark the loop complete.
        assert all(not event["until_reached"] for event in progress_events[:-1])

    @pytest.mark.asyncio
    async def test_loop_start_from_missing_cell_raises_clear_error(self, tmp_path):
        """A ``start_from`` pointing at a non-existent iteration fails cleanly."""
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        notebook_dir = create_notebook(tmp_path, "loop_fork_missing")
        add_cell_to_notebook(notebook_dir, "loop")
        write_cell(
            notebook_dir,
            "loop",
            (
                "# @loop max_iter=1 carry=state start_from=ghost@iter=0\n"
                "state = {'n': state['n'] + 1}\n"
            ),
        )

        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        executor = CellExecutor(session)

        result = await executor.execute_cell(
            "loop",
            (
                "# @loop max_iter=1 carry=state start_from=ghost@iter=0\n"
                "state = {'n': state['n'] + 1}\n"
            ),
        )
        assert not result.success
        assert "seed" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_running_the_loop_again_does_not_rerun_the_body(self, loop_notebook, tmp_path):
        """An unchanged loop cell is served from the cache like any other.

        The body appends a character per iteration to a file, so the count
        says whether the loop actually ran rather than whether the executor
        claimed it did.
        """
        from strata.notebook.writer import write_cell

        notebook_dir, session = loop_notebook
        trace = tmp_path / "trace.txt"
        loop_source = (
            "# @loop max_iter=3 carry=state\n"
            f"open({str(trace)!r}, 'a').write('x')\n"
            "state = {'n': state['n'] + 1, 'history': state['history'] + [state['n']]}\n"
        )
        write_cell(notebook_dir, "loop", loop_source)
        session.reload()

        executor = CellExecutor(session)
        await executor.execute_cell("seed", "state = {'n': 0, 'history': []}")

        first = await executor.execute_cell("loop", loop_source)
        assert first.success, first.error
        assert trace.read_text() == "xxx"

        second = await executor.execute_cell("loop", loop_source)
        assert second.success, second.error
        assert second.cache_hit
        assert second.execution_method == "cached"
        assert second.artifact_uri == first.artifact_uri
        assert trace.read_text() == "xxx"

    @pytest.mark.asyncio
    async def test_editing_the_loop_body_runs_it_again(self, loop_notebook, tmp_path):
        """The cache is keyed on provenance, so an edited body misses and the loop runs again."""
        from strata.notebook.writer import write_cell

        notebook_dir, session = loop_notebook
        trace = tmp_path / "trace.txt"
        body = f"open({str(trace)!r}, 'a').write('x')\n"
        first_source = (
            "# @loop max_iter=2 carry=state\n"
            + body
            + "state = {'n': state['n'] + 1, 'history': []}\n"
        )
        write_cell(notebook_dir, "loop", first_source)
        session.reload()

        executor = CellExecutor(session)
        await executor.execute_cell("seed", "state = {'n': 0, 'history': []}")
        first = await executor.execute_cell("loop", first_source)
        assert first.success, first.error
        assert trace.read_text() == "xx"

        edited_source = (
            "# @loop max_iter=2 carry=state\n"
            + body
            + "state = {'n': state['n'] + 100, 'history': []}\n"
        )
        write_cell(notebook_dir, "loop", edited_source)
        session.reload()
        second = await executor.execute_cell("loop", edited_source)

        assert second.success, second.error
        assert not second.cache_hit
        assert trace.read_text() == "xxxx"

        artifact_mgr = session.get_artifact_manager()
        final = json.loads(artifact_mgr.load_iteration_blob("loop", "state", 1))
        assert final == {"n": 200, "history": []}


class TestExecuteCellMaterializesUpstreams:
    @pytest.mark.asyncio
    async def test_upstreams_are_materialized(self, sample_notebook, monkeypatch):
        from strata.notebook.writer import write_cell

        write_cell(sample_notebook.path, "cell1", "x = 41")
        write_cell(sample_notebook.path, "cell2", "y = x + 1")
        sample_notebook.reload()

        executor = CellExecutor(sample_notebook)
        materialize_calls: list[str] = []
        original = executor._materialize_upstreams

        async def spy(cell_id: str) -> None:
            materialize_calls.append(cell_id)
            await original(cell_id)

        monkeypatch.setattr(executor, "_materialize_upstreams", spy)

        await executor.execute_cell("cell2", "y = x + 1")

        # The recursion calls _materialize_upstreams for cell2 (target), then for
        # cell1 (its upstream).
        assert "cell2" in materialize_calls, (
            "Default execute_cell should call _materialize_upstreams for the target cell, "
            f"got {materialize_calls}"
        )
        assert "cell1" in materialize_calls, (
            "Default execute_cell should recurse into upstream materialization, "
            f"got {materialize_calls}"
        )


class TestMountCredentialsPassthrough:
    """The mount-credentials kwarg threads through to ``MountResolver``."""

    def test_default_is_empty(self, sample_notebook):
        executor = CellExecutor(sample_notebook)
        assert executor._mount_resolver.credentials == {}

    def test_kwarg_reaches_resolver(self, sample_notebook):
        """The kwarg lands on the resolver verbatim."""
        creds = {
            "s3": {"endpoint_url": "http://minio:9000", "anon": True},
            "az": {"connection_string": "DefaultEndpointsProtocol=http;..."},
        }
        executor = CellExecutor(sample_notebook, mount_credentials=creds)
        assert executor._mount_resolver.credentials == creds


class TestAmbientRemoteStore:
    """The ambient `strata` client can target a remote shared store."""

    def test_ambient_url_and_headers_default_local(self):
        from unittest.mock import MagicMock

        from strata.notebook.executor import CellExecutor

        ex = MagicMock()
        cfg = MagicMock(server_url="http://local:8765")
        cfg.notebook_remote_store_url = None
        ex._lake_config.return_value = cfg

        assert CellExecutor._ambient_strata_url(ex) == "http://local:8765"
        assert CellExecutor._ambient_strata_headers(ex) == {}

    def test_ambient_url_and_headers_remote(self):
        from unittest.mock import MagicMock

        from strata.notebook.executor import CellExecutor

        ex = MagicMock()
        cfg = MagicMock(
            server_url="http://local:8765",
            notebook_remote_store_url="https://store.team.example",
            notebook_remote_store_headers={"X-Tenant-ID": "team-a", "X-Strata-Principal": "alice"},
        )
        ex._lake_config.return_value = cfg

        assert CellExecutor._ambient_strata_url(ex) == "https://store.team.example"
        assert CellExecutor._ambient_strata_headers(ex) == {
            "X-Tenant-ID": "team-a",
            "X-Strata-Principal": "alice",
        }


class TestNoInterpreterRunsNoCellCode:
    """A session without an interpreter (none recorded yet, or its sync raised) runs no cell code.

    Falling back to ``python`` on PATH would store another environment's results under this
    notebook's provenance.
    """

    @staticmethod
    def _notebook(tmp_path, marker: Path):
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        notebook_dir = create_notebook(tmp_path, "no_interpreter")
        add_cell_to_notebook(notebook_dir, "seed")
        write_cell(notebook_dir, "seed", "state = 0")
        add_cell_to_notebook(notebook_dir, "c1", after_cell_id="seed")
        write_cell(notebook_dir, "c1", f"open({str(marker)!r}, 'w').close()\nx = 1\n")
        return notebook_dir, NotebookSession(parse_notebook(notebook_dir), notebook_dir)

    @pytest.mark.asyncio
    async def test_a_cell_is_refused(self, tmp_path):
        marker = tmp_path / "ran"
        _, session = self._notebook(tmp_path, marker)
        assert session.venv_python is None

        source = session.notebook_state.cells[1].source
        result = await CellExecutor(session).execute_cell("c1", source)

        assert not result.success
        assert "not ready" in (result.error or "")
        assert not marker.exists()

    @pytest.mark.asyncio
    async def test_a_loop_iteration_is_refused(self, tmp_path):
        from strata.notebook.writer import write_cell

        marker = tmp_path / "ran"
        notebook_dir, session = self._notebook(tmp_path, marker)
        loop_source = (
            f"# @loop max_iter=2 carry=state\nopen({str(marker)!r}, 'w').close()\n"
            "state = state + 1\n"
        )
        write_cell(notebook_dir, "c1", loop_source)
        session.reload()
        session.refresh_environment_runtime()
        executor = CellExecutor(session)
        assert (await executor.execute_cell("seed", "state = 0")).success

        session.venv_python = None
        result = await executor.execute_cell("c1", loop_source)

        assert not result.success
        assert "not ready" in (result.error or "")
        assert not marker.exists()

    @pytest.mark.asyncio
    async def test_a_batch_is_refused(self, tmp_path):
        marker = tmp_path / "ran"
        _, session = self._notebook(tmp_path, marker)
        spec = {
            "cell_id": "c1",
            "source": session.notebook_state.cells[1].source,
            "consumed_vars": [],
            "env": {},
            "mount_manifest": {},
            "source_hash": "",
            "env_hash": "",
        }

        result = await CellExecutor(session).execute_batch([spec])

        assert not result.completed
        assert result.cell_results[0].status == "cell_error"
        assert "not ready" in (result.cell_results[0].error or "")
        assert not marker.exists()

    @pytest.mark.asyncio
    async def test_cell_tests_are_refused(self, tmp_path):
        marker = tmp_path / "ran"
        _, session = self._notebook(tmp_path, marker)

        with pytest.raises(RuntimeError, match="not ready"):
            await CellExecutor(session).run_cell_tests(
                "c1", "def test_x(cell):\n    assert cell.x == 1\n"
            )
        assert not marker.exists()
