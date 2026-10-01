"""Markdown cells are pure prose: no execution, no DAG edges, no provenance chain."""

from __future__ import annotations

import asyncio

import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    write_cell,
)


class TestMarkdownCellPersistence:
    def test_add_markdown_cell_creates_md_file(self, tmp_path):
        nb_dir = create_notebook(tmp_path, "md_persist")
        add_cell_to_notebook(nb_dir, "abc12345", language="markdown")

        cells_dir = nb_dir / "cells"
        assert (cells_dir / "abc12345.md").exists()
        assert not (cells_dir / "abc12345.py").exists()

    def test_python_cell_still_uses_py_file(self, tmp_path):
        nb_dir = create_notebook(tmp_path, "py_persist")
        add_cell_to_notebook(nb_dir, "py_cell1", language="python")
        assert (nb_dir / "cells" / "py_cell1.py").exists()

    def test_parse_round_trip_preserves_markdown_source(self, tmp_path):
        nb_dir = create_notebook(tmp_path, "md_roundtrip")
        add_cell_to_notebook(nb_dir, "doc01234", language="markdown")
        write_cell(nb_dir, "doc01234", "# Hello\n\nMarkdown body.")

        state = parse_notebook(nb_dir)
        cell = next(c for c in state.cells if c.id == "doc01234")
        assert cell.language == "markdown"
        assert cell.source == "# Hello\n\nMarkdown body."


class TestMarkdownCellAnalysis:
    def test_markdown_has_no_defines_or_references(self, tmp_path):
        nb_dir = create_notebook(tmp_path, "md_dag")
        # Source that *would* parse as Python with defines+references: treated as
        # Python, it would record ``x`` as a define.
        add_cell_to_notebook(nb_dir, "p_cell", language="python")
        write_cell(nb_dir, "p_cell", "x = 1")

        add_cell_to_notebook(nb_dir, "md_cell", language="markdown")
        write_cell(nb_dir, "md_cell", "# Doc\n\nx = 1 here is just prose.")

        session = NotebookSession(parse_notebook(nb_dir), nb_dir)
        md_cell = next(c for c in session.notebook_state.cells if c.id == "md_cell")
        assert md_cell.defines == []
        assert md_cell.references == []
        # The python cell's ``x`` stays visible in the DAG producer map; the markdown
        # cell must not shadow it.
        assert session.dag is not None
        assert session.dag.variable_producer.get("x") == "p_cell"


class TestMarkdownCellExecution:
    def test_executor_short_circuits_with_no_output(self, tmp_path):
        """``execute_cell`` returns success with no display outputs.

        The frontend already renders the source in place; an output would duplicate it.
        """
        nb_dir = create_notebook(tmp_path, "md_exec")
        add_cell_to_notebook(nb_dir, "md_cell", language="markdown")
        body = "# Hi\n\nThis is **bold**."
        write_cell(nb_dir, "md_cell", body)

        session = NotebookSession(parse_notebook(nb_dir), nb_dir)
        executor = CellExecutor(session, session.warm_pool)

        result = asyncio.run(executor.execute_cell("md_cell", body))

        assert result.success is True
        # cache_hit is the structural tell that the fast path ran (no uv
        # subprocess spawned), not a wall-clock measurement.
        assert result.cache_hit is True
        assert result.display_outputs == []
        assert result.display_output is None
        # start_time is wall-clock, so the markdown branch must subtract via
        # ``time.time()``, not ``time.monotonic()``; mixing them yields an epoch-scale
        # or negative duration. 60s is far above any real render, far below that bug.
        assert 0 <= result.duration_ms < 60_000


class TestHarnessCrashDiagnostic:
    """When the harness dies on import, the executor surfaces its stderr.

    A manifest without a ``success`` key is the leftover input, not a result;
    reading it as one gave a generic "Unknown error" for missing harness deps.
    """

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_harness_returns_meaningful_error_when_subprocess_crashes(
        self, tmp_path, monkeypatch
    ):
        import asyncio
        import json

        from strata.notebook.executor import CellExecutor

        # Hand-craft an "output_dir" containing only an input-shaped
        # manifest.json (no ``success`` field) to simulate a harness that
        # exited before writing its result.
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps({"source": "x = 1", "inputs": {}, "output_dir": str(output_dir)})
        )

        # ``_run_harness`` only uses ``self.harness_path`` and ``self.session.path``,
        # so a minimal stand-in is enough.
        class _FakeSession:
            path = output_dir

        executor = CellExecutor.__new__(CellExecutor)
        executor.harness_path = output_dir / "harness.py"  # not actually executed
        executor.session = _FakeSession()  # type: ignore[assignment]

        class _FakeProc:
            pid = 0

            async def communicate(self):
                return (
                    b"",
                    b"Traceback (most recent call last):\n"
                    b'  File "harness.py", line 1, in <module>\n'
                    b"    import orjson\n"
                    b"ModuleNotFoundError: No module named 'orjson'\n",
                )

            def kill(self):
                pass

            async def wait(self):
                return None

        async def fake_subprocess_exec(*args, **kwargs):
            return _FakeProc()

        # Use monkeypatch so cleanup is automatic and pytest's event loop
        # plumbing isn't confused by manual reassignment.
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess_exec)

        result = await executor._run_harness(manifest_path, output_dir, 5.0)

        assert result["success"] is False
        assert "ModuleNotFoundError" in result["error"]
        assert "orjson" in result["error"]
        # stderr should also be preserved verbatim for the UI's stderr panel.
        assert "ModuleNotFoundError" in result["stderr"]


class TestMarkdownCellStaleness:
    def test_markdown_cells_are_always_ready(self, tmp_path):
        nb_dir = create_notebook(tmp_path, "md_stale")
        add_cell_to_notebook(nb_dir, "md_cell", language="markdown")
        write_cell(nb_dir, "md_cell", "# Doc")

        session = NotebookSession(parse_notebook(nb_dir), nb_dir)
        staleness = session.compute_staleness()
        from strata.notebook.models import CellStatus

        assert staleness["md_cell"].status == CellStatus.READY
