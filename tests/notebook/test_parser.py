"""Tests for notebook parser."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from strata.notebook.models import (
    CellMeta,
    MountMode,
    MountSpec,
    NotebookToml,
    WorkerBackendType,
    WorkerSpec,
)
from strata.notebook.parser import parse_notebook
from strata.notebook.writer import create_notebook, write_cell, write_notebook_toml


def test_parse_empty_notebook():
    """Test parsing a notebook with no cells."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a notebook
        notebook_dir = create_notebook(tmpdir_path, "Test Notebook")

        # Parse it
        notebook_state = parse_notebook(notebook_dir)

        assert isinstance(notebook_state.id, str) and len(notebook_state.id) > 0
        assert notebook_state.name == "Test Notebook"
        assert notebook_state.cells == []
        assert notebook_state.path == notebook_dir


def test_parse_notebook_with_cells():
    """Test parsing a notebook with multiple cells."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a notebook
        notebook_dir = create_notebook(tmpdir_path, "Multi-cell Notebook")

        # Add cells
        from strata.notebook.writer import add_cell_to_notebook

        cell1_id = "cell-1"
        add_cell_to_notebook(notebook_dir, cell1_id)

        cell2_id = "cell-2"
        add_cell_to_notebook(notebook_dir, cell2_id, after_cell_id=cell1_id)

        # Write source for cells
        write_cell(notebook_dir, cell1_id, "x = 1 + 1")
        write_cell(notebook_dir, cell2_id, "y = x * 2")

        # Parse it
        notebook_state = parse_notebook(notebook_dir)

        assert notebook_state.name == "Multi-cell Notebook"
        assert len(notebook_state.cells) == 2
        assert notebook_state.cells[0].id == cell1_id
        assert notebook_state.cells[0].source == "x = 1 + 1"
        assert notebook_state.cells[1].id == cell2_id
        assert notebook_state.cells[1].source == "y = x * 2"


def test_parse_notebook_missing_cells_directory():
    """Test parsing a notebook with missing cell files (graceful degradation)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create notebook directory structure manually
        notebook_dir = tmpdir_path / "test_notebook"
        notebook_dir.mkdir()
        cells_dir = notebook_dir / "cells"
        cells_dir.mkdir()

        # Create notebook.toml with cell reference
        now = datetime.now(tz=UTC)
        notebook_toml = NotebookToml(
            notebook_id="test-123",
            name="Missing Cell Notebook",
            created_at=now,
            updated_at=now,
            cells=[CellMeta(id="cell-1", file="missing.py", language="python", order=0)],
        )
        write_notebook_toml(notebook_dir, notebook_toml)

        # Note: we don't create the actual cell file

        # Parse it - should gracefully handle missing file
        notebook_state = parse_notebook(notebook_dir)

        assert notebook_state.name == "Missing Cell Notebook"
        assert len(notebook_state.cells) == 1
        assert notebook_state.cells[0].id == "cell-1"
        assert notebook_state.cells[0].source == ""  # Empty source for missing file


def test_parse_notebook_not_found():
    """Test parsing a non-existent notebook."""
    with pytest.raises(FileNotFoundError):
        parse_notebook(Path("/nonexistent/notebook"))


def test_parse_and_reload_after_edit():
    """Test round-trip: parse, edit, reload."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create notebook
        notebook_dir = create_notebook(tmpdir_path, "Editable Notebook")

        from strata.notebook.writer import add_cell_to_notebook

        cell_id = "cell-1"
        add_cell_to_notebook(notebook_dir, cell_id)
        write_cell(notebook_dir, cell_id, "original code")

        # Parse it
        notebook_state1 = parse_notebook(notebook_dir)
        assert notebook_state1.cells[0].source == "original code"

        # Edit cell
        write_cell(notebook_dir, cell_id, "modified code")

        # Reload
        notebook_state2 = parse_notebook(notebook_dir)
        assert notebook_state2.cells[0].source == "modified code"


def test_parse_notebook_merges_notebook_and_cell_mounts():
    """Cell state should include notebook defaults plus cell-level overrides."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = Path(tmpdir) / "mount_notebook"
        notebook_dir.mkdir()
        (notebook_dir / "cells").mkdir()

        now = datetime.now(tz=UTC)
        notebook_toml = NotebookToml(
            notebook_id="test-mounts",
            name="Mount Notebook",
            created_at=now,
            updated_at=now,
            mounts=[
                MountSpec(name="raw_data", uri="s3://bucket/raw", mode=MountMode.READ_ONLY),
                MountSpec(name="scratch", uri="file:///tmp/base", mode=MountMode.READ_WRITE),
            ],
            cells=[
                CellMeta(
                    id="cell-1",
                    file="cell1.py",
                    language="python",
                    order=0,
                    mounts=[
                        MountSpec(
                            name="scratch",
                            uri="file:///tmp/override",
                            mode=MountMode.READ_WRITE,
                        )
                    ],
                )
            ],
        )
        write_notebook_toml(notebook_dir, notebook_toml)

        notebook_state = parse_notebook(notebook_dir)
        assert len(notebook_state.mounts) == 2
        mounts = {mount.name: mount for mount in notebook_state.cells[0].mounts}
        overrides = {mount.name: mount for mount in notebook_state.cells[0].mount_overrides}

        assert set(mounts) == {"raw_data", "scratch"}
        assert mounts["raw_data"].uri == "s3://bucket/raw"
        assert mounts["scratch"].uri == "file:///tmp/override"
        assert set(overrides) == {"scratch"}


def test_parse_notebook_resolves_notebook_and_cell_workers():
    """Cell state should include notebook worker defaults plus cell overrides."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = Path(tmpdir) / "worker_notebook"
        notebook_dir.mkdir()
        (notebook_dir / "cells").mkdir()

        now = datetime.now(tz=UTC)
        notebook_toml = NotebookToml(
            notebook_id="test-workers",
            name="Worker Notebook",
            created_at=now,
            updated_at=now,
            worker="gpu-default",
            cells=[
                CellMeta(
                    id="cell-1",
                    file="cell1.py",
                    language="python",
                    order=0,
                ),
                CellMeta(
                    id="cell-2",
                    file="cell2.py",
                    language="python",
                    order=1,
                    worker="gpu-override",
                ),
            ],
        )
        write_notebook_toml(notebook_dir, notebook_toml)

        notebook_state = parse_notebook(notebook_dir)

        assert notebook_state.worker == "gpu-default"
        assert notebook_state.cells[0].worker == "gpu-default"
        assert notebook_state.cells[0].worker_override is None
        assert notebook_state.cells[1].worker == "gpu-override"
        assert notebook_state.cells[1].worker_override == "gpu-override"


def test_parse_notebook_preserves_worker_registry():
    """Notebook-scoped worker definitions should round-trip through parsing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = Path(tmpdir) / "worker_registry"
        notebook_dir.mkdir()
        (notebook_dir / "cells").mkdir()

        now = datetime.now(tz=UTC)
        notebook_toml = NotebookToml(
            notebook_id="test-worker-registry",
            name="Worker Registry Notebook",
            created_at=now,
            updated_at=now,
            workers=[
                WorkerSpec(name="local", backend=WorkerBackendType.LOCAL),
                WorkerSpec(
                    name="gpu-a100",
                    backend=WorkerBackendType.EXECUTOR,
                    runtime_id="cuda-12.4",
                    config={"url": "https://executor.internal/gpu-a100"},
                ),
            ],
            cells=[],
        )
        write_notebook_toml(notebook_dir, notebook_toml)

        notebook_state = parse_notebook(notebook_dir)

        assert [worker.name for worker in notebook_state.workers] == [
            "local",
            "gpu-a100",
        ]
        assert notebook_state.workers[1].backend == WorkerBackendType.EXECUTOR
        assert notebook_state.workers[1].runtime_id == "cuda-12.4"
        assert notebook_state.workers[1].config.url == "https://executor.internal/gpu-a100"


def test_parse_notebook_resolves_notebook_and_cell_runtime_settings():
    """Cell state should include notebook timeout/env defaults plus cell overrides."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = Path(tmpdir) / "runtime_notebook"
        notebook_dir.mkdir()
        (notebook_dir / "cells").mkdir()

        now = datetime.now(tz=UTC)
        notebook_toml = NotebookToml(
            notebook_id="test-runtime",
            name="Runtime Notebook",
            created_at=now,
            updated_at=now,
            timeout=12.5,
            env={"API_ROOT": "https://example.test", "APP_MODE": "base"},
            cells=[
                CellMeta(
                    id="cell-1",
                    file="cell1.py",
                    language="python",
                    order=0,
                ),
                CellMeta(
                    id="cell-2",
                    file="cell2.py",
                    language="python",
                    order=1,
                    timeout=3.0,
                    env={"APP_MODE": "override"},
                ),
            ],
        )
        write_notebook_toml(notebook_dir, notebook_toml)

        notebook_state = parse_notebook(notebook_dir)

        assert notebook_state.timeout == 12.5
        assert notebook_state.env == {
            "API_ROOT": "https://example.test",
            "APP_MODE": "base",
        }
        assert notebook_state.cells[0].timeout == 12.5
        assert notebook_state.cells[0].timeout_override is None
        assert notebook_state.cells[0].env == {
            "API_ROOT": "https://example.test",
            "APP_MODE": "base",
        }
        assert notebook_state.cells[1].timeout == 3.0
        assert notebook_state.cells[1].timeout_override == 3.0
        assert notebook_state.cells[1].env == {
            "API_ROOT": "https://example.test",
            "APP_MODE": "override",
        }
        assert notebook_state.cells[1].env_overrides == {"APP_MODE": "override"}
