"""Tests for notebook writer."""

import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from strata.notebook import writer as writer_module
from strata.notebook.models import (
    CellMeta,
    MountMode,
    MountSpec,
    NotebookToml,
    WorkerBackendType,
    WorkerSpec,
)
from strata.notebook.parser import parse_notebook
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    remove_cell_from_notebook,
    remove_variant_group_entry,
    rename_notebook,
    reorder_cells,
    set_variant_active,
    update_environment_metadata,
    update_notebook_connections,
    update_notebook_env,
    update_notebook_timeout,
    update_notebook_worker,
    update_notebook_workers,
    write_cell,
    write_notebook_toml,
)


def test_create_notebook():
    """Test creating a new notebook."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create notebook
        notebook_dir = create_notebook(tmpdir_path, "New Notebook")

        # Verify structure
        assert notebook_dir.exists()
        assert (notebook_dir / "notebook.toml").exists()
        assert (notebook_dir / "pyproject.toml").exists()
        assert (notebook_dir / "cells").exists()
        assert (notebook_dir / "cells").is_dir()


def test_update_environment_metadata_reads_pyvenv_cfg_without_subprocess(
    monkeypatch: pytest.MonkeyPatch,
):
    """Refreshing environment metadata should reuse pyvenv.cfg when available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Metadata Probe Test")

        def fail_subprocess(*args, **kwargs):
            raise AssertionError("venv python probe should not spawn a subprocess")

        monkeypatch.setattr(
            writer_module,
            "read_venv_runtime_python_version",
            lambda *_args, **_kwargs: "3.13.3",
        )
        monkeypatch.setattr(writer_module.subprocess, "run", fail_subprocess)

        update_environment_metadata(notebook_dir)

        from strata.notebook.runtime_state import load_runtime_state

        environment = load_runtime_state(notebook_dir).get("environment", {})
        assert environment.get("runtime_python_version") == "3.13.3"


def test_write_cell():
    """Test writing cell source."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create notebook
        notebook_dir = create_notebook(tmpdir_path, "Cell Write Test")

        # Add a cell
        cell_id = "test-cell"
        add_cell_to_notebook(notebook_dir, cell_id)

        # Write source
        source = "x = 1 + 1\ny = x * 2"
        write_cell(notebook_dir, cell_id, source)

        # Verify file was written
        cells_dir = notebook_dir / "cells"
        cell_file = cells_dir / f"{cell_id}.py"
        assert cell_file.exists()
        assert cell_file.read_text() == source


def test_write_cell_not_found():
    """Test writing to a non-existent cell."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create notebook
        notebook_dir = create_notebook(tmpdir_path, "Cell Write Test")

        # Try to write to non-existent cell
        with pytest.raises(FileNotFoundError, match="Cell .* not found"):
            write_cell(notebook_dir, "nonexistent", "code")


def test_add_cell():
    """Test adding cells."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create notebook
        notebook_dir = create_notebook(tmpdir_path, "Add Cell Test")

        # Add first cell
        cell1_id = "cell-1"
        add_cell_to_notebook(notebook_dir, cell1_id)

        # Add second cell
        cell2_id = "cell-2"
        add_cell_to_notebook(notebook_dir, cell2_id)

        # Parse and verify
        notebook_state = parse_notebook(notebook_dir)
        assert len(notebook_state.cells) == 2
        assert notebook_state.cells[0].id == cell1_id
        assert notebook_state.cells[1].id == cell2_id


def test_add_cell_after():
    """Test adding cell after a specific cell."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create notebook
        notebook_dir = create_notebook(tmpdir_path, "Add Cell After Test")

        # Add cells
        cell1_id = "cell-1"
        add_cell_to_notebook(notebook_dir, cell1_id)

        cell2_id = "cell-2"
        add_cell_to_notebook(notebook_dir, cell2_id)

        # Add cell after cell1
        cell1_5_id = "cell-1.5"
        add_cell_to_notebook(notebook_dir, cell1_5_id, after_cell_id=cell1_id)

        # Parse and verify order
        notebook_state = parse_notebook(notebook_dir)
        assert len(notebook_state.cells) == 3
        cell_ids = [c.id for c in notebook_state.cells]
        assert cell_ids.index(cell1_id) < cell_ids.index(cell1_5_id)
        assert cell_ids.index(cell1_5_id) < cell_ids.index(cell2_id)


def test_remove_cell():
    """Test removing cells."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create notebook
        notebook_dir = create_notebook(tmpdir_path, "Remove Cell Test")

        # Add cells
        cell1_id = "cell-1"
        add_cell_to_notebook(notebook_dir, cell1_id)

        cell2_id = "cell-2"
        add_cell_to_notebook(notebook_dir, cell2_id)

        # Remove first cell
        remove_cell_from_notebook(notebook_dir, cell1_id)

        # Verify
        notebook_state = parse_notebook(notebook_dir)
        assert len(notebook_state.cells) == 1
        assert notebook_state.cells[0].id == cell2_id


def test_remove_cell_not_found():
    """Test removing a non-existent cell."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create notebook
        notebook_dir = create_notebook(tmpdir_path, "Remove Cell Test")

        # Try to remove non-existent cell
        with pytest.raises(FileNotFoundError, match="Cell .* not found"):
            remove_cell_from_notebook(notebook_dir, "nonexistent")


def test_reorder_cells():
    """Test reordering cells."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create notebook
        notebook_dir = create_notebook(tmpdir_path, "Reorder Test")

        # Add cells
        cell1_id = "cell-1"
        add_cell_to_notebook(notebook_dir, cell1_id)

        cell2_id = "cell-2"
        add_cell_to_notebook(notebook_dir, cell2_id)

        cell3_id = "cell-3"
        add_cell_to_notebook(notebook_dir, cell3_id)

        # Reorder to [2, 3, 1]
        reorder_cells(notebook_dir, [cell2_id, cell3_id, cell1_id])

        # Verify
        notebook_state = parse_notebook(notebook_dir)
        cell_ids = [c.id for c in notebook_state.cells]
        assert cell_ids == [cell2_id, cell3_id, cell1_id]


def test_rename_notebook():
    """Test renaming a notebook."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create notebook
        notebook_dir = create_notebook(tmpdir_path, "Original Name")

        # Rename
        rename_notebook(notebook_dir, "New Name")

        # Verify
        notebook_state = parse_notebook(notebook_dir)
        assert notebook_state.name == "New Name"


def test_write_notebook_toml():
    """Test writing notebook.toml."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create notebook
        notebook_dir = create_notebook(tmpdir_path, "TOML Test")

        # Create a NotebookToml
        now = datetime.now(tz=UTC)
        notebook_toml = NotebookToml(
            notebook_id="custom-id",
            name="Custom Notebook",
            created_at=now,
            updated_at=now,
            worker="gpu-default",
            timeout=9.5,
            env={"API_ROOT": "https://example.test"},
            ai={"model": "gpt-4o", "base_url": "https://api.openai.com/v1"},
            mounts=[
                MountSpec(name="raw_data", uri="s3://bucket/dataset", mode=MountMode.READ_ONLY),
            ],
            cells=[
                CellMeta(
                    id="c1",
                    file="cell1.py",
                    language="python",
                    order=0,
                    worker="gpu-worker",
                    timeout=2.0,
                    env={"CELL_MODE": "cell-secret"},
                    mounts=[
                        MountSpec(
                            name="scratch",
                            uri="file:///tmp/scratch",
                            mode=MountMode.READ_WRITE,
                        )
                    ],
                ),
                CellMeta(id="c2", file="cell2.py", language="python", order=1),
            ],
        )

        # Write it
        write_notebook_toml(notebook_dir, notebook_toml)

        # Verify by reading it back
        notebook_state = parse_notebook(notebook_dir)
        assert notebook_state.id == "custom-id"
        assert notebook_state.name == "Custom Notebook"
        assert notebook_state.worker == "gpu-default"
        assert notebook_state.timeout == 9.5
        assert notebook_state.env == {"API_ROOT": "https://example.test"}
        assert len(notebook_state.cells) == 2
        assert notebook_state.cells[0].worker == "gpu-worker"
        assert notebook_state.cells[0].worker_override == "gpu-worker"
        assert notebook_state.cells[0].timeout == 2.0
        assert notebook_state.cells[0].timeout_override == 2.0
        assert notebook_state.cells[0].env == {
            "API_ROOT": "https://example.test",
            "CELL_MODE": "cell-secret",
        }
        assert notebook_state.cells[0].env_overrides == {"CELL_MODE": "cell-secret"}
        assert notebook_state.cells[1].worker == "gpu-default"
        assert notebook_state.cells[1].timeout == 9.5
        assert len(notebook_state.cells[0].mounts) == 2
        assert {mount.name for mount in notebook_state.cells[0].mounts} == {
            "raw_data",
            "scratch",
        }
        assert len(notebook_state.cells[1].mounts) == 1
        assert notebook_state.cells[1].mounts[0].name == "raw_data"

        with open(notebook_dir / "notebook.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["ai"] == {
            "model": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
        }


def test_update_notebook_worker():
    """Test persisting notebook-level worker configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Worker Notebook")
        update_notebook_worker(notebook_dir, "gpu-default")

        notebook_state = parse_notebook(notebook_dir)
        assert notebook_state.worker == "gpu-default"


def test_update_notebook_workers():
    """Test persisting notebook-scoped worker definitions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Worker Catalog Notebook")
        update_notebook_workers(
            notebook_dir,
            [
                WorkerSpec(name="local", backend=WorkerBackendType.LOCAL),
                WorkerSpec(
                    name="gpu-a100",
                    backend=WorkerBackendType.EXECUTOR,
                    runtime_id="cuda-12.4",
                    config={"url": "https://executor.internal/gpu-a100"},
                ),
            ],
        )

        notebook_state = parse_notebook(notebook_dir)
        assert [worker.name for worker in notebook_state.workers] == [
            "local",
            "gpu-a100",
        ]
        assert notebook_state.workers[1].backend == WorkerBackendType.EXECUTOR
        assert notebook_state.workers[1].runtime_id == "cuda-12.4"
        assert notebook_state.workers[1].config == {"url": "https://executor.internal/gpu-a100"}


def test_update_notebook_timeout_and_env():
    """Test persisting notebook-level timeout/env configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Notebook Runtime")
        update_notebook_timeout(notebook_dir, 7.5)
        update_notebook_env(notebook_dir, {"DATABASE_URL": "postgres://localhost/db"})

        notebook_state = parse_notebook(notebook_dir)
        assert notebook_state.timeout == 7.5
        assert notebook_state.env == {"DATABASE_URL": "postgres://localhost/db"}


def test_update_notebook_env_preserves_ai_config():
    """Notebook runtime edits should not strip [ai] configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Notebook AI Runtime")

        with open(notebook_dir / "notebook.toml", "a", encoding="utf-8") as f:
            f.write('\n[ai]\nmodel = "gpt-4o"\nbase_url = "https://api.openai.com/v1"\n')

        update_notebook_env(notebook_dir, {"DATABASE_URL": "postgres://localhost/db"})

        with open(notebook_dir / "notebook.toml", "rb") as f:
            data = tomllib.load(f)

        assert data["env"] == {"DATABASE_URL": "postgres://localhost/db"}
        assert data["ai"] == {
            "model": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
        }


def test_sensitive_only_env_block_is_not_persisted():
    """When every env entry is a blanked sensitive key, skip the block.

    Earlier behavior left ``[env]\nOPENAI_API_KEY = ""`` in the
    committed notebook.toml — noise for shared/example notebooks with
    no real config value persisted.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Sensitive Only Test")

        update_notebook_env(notebook_dir, {"OPENAI_API_KEY": "sk-proj-secret"})

        with open(notebook_dir / "notebook.toml", "rb") as f:
            data = tomllib.load(f)

        assert "env" not in data


def test_update_notebook_writers_are_no_op_when_value_unchanged():
    """The write-if-changed invariant across update_notebook_* writers.

    Each of these touches notebook.toml and bumps updated_at — but
    only when the persisted value actually changed. A redundant call
    with the current value should leave the file byte-identical.
    """
    from strata.notebook.writer import (
        rename_notebook,
        update_notebook_ai_model,
        update_notebook_mounts,
        update_notebook_timeout,
        update_notebook_worker,
        update_notebook_workers,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Write Once Test")
        notebook_toml = notebook_dir / "notebook.toml"

        # Seed with non-default values so subsequent identical writes
        # actually exercise the equality path.
        update_notebook_worker(notebook_dir, "gpu-a100")
        update_notebook_timeout(notebook_dir, 30.0)
        update_notebook_ai_model(notebook_dir, "gpt-4o")
        rename_notebook(notebook_dir, "Write Once Renamed")
        update_notebook_mounts(
            notebook_dir,
            [MountSpec(name="data", uri="s3://bucket/prefix", mode=MountMode.READ_ONLY)],
        )
        update_notebook_workers(
            notebook_dir,
            [WorkerSpec(name="local", backend=WorkerBackendType.LOCAL)],
        )

        snapshot = notebook_toml.read_bytes()

        # Second call with the exact same value is a no-op.
        update_notebook_worker(notebook_dir, "gpu-a100")
        update_notebook_timeout(notebook_dir, 30.0)
        update_notebook_ai_model(notebook_dir, "gpt-4o")
        rename_notebook(notebook_dir, "Write Once Renamed")
        update_notebook_mounts(
            notebook_dir,
            [MountSpec(name="data", uri="s3://bucket/prefix", mode=MountMode.READ_ONLY)],
        )
        update_notebook_workers(
            notebook_dir,
            [WorkerSpec(name="local", backend=WorkerBackendType.LOCAL)],
        )

        assert notebook_toml.read_bytes() == snapshot, (
            "repeated writes with identical values should not change notebook.toml"
        )

        # Sanity: an actual change still bumps updated_at / rewrites.
        update_notebook_timeout(notebook_dir, 60.0)
        assert notebook_toml.read_bytes() != snapshot


def test_sensitive_only_env_is_no_op_no_updated_at_bump():
    """Typing an API key in the Runtime panel shouldn't churn
    notebook.toml: no persistable change → no rewrite → no updated_at
    bump. Otherwise examples get git diffs for invisible edits.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "No Churn Test")
        notebook_toml = notebook_dir / "notebook.toml"

        before = notebook_toml.read_bytes()
        update_notebook_env(notebook_dir, {"OPENAI_API_KEY": "sk-proj-secret"})
        assert notebook_toml.read_bytes() == before

        # Second call with a different sensitive-only value is also a no-op —
        # the persisted shape is identical.
        update_notebook_env(notebook_dir, {"ANTHROPIC_API_KEY": "sk-ant-other"})
        assert notebook_toml.read_bytes() == before


def test_env_block_persists_when_mixed_with_non_sensitive():
    """A sensitive key alongside any non-sensitive value keeps the slot."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Mixed Env Test")

        update_notebook_env(
            notebook_dir,
            {"OPENAI_API_KEY": "sk-proj-secret", "DATABASE_URL": "postgres://x"},
        )

        with open(notebook_dir / "notebook.toml", "rb") as f:
            data = tomllib.load(f)

        # Sensitive slot preserved as a blanked reminder; non-sensitive
        # value kept verbatim.
        assert data["env"]["OPENAI_API_KEY"] == ""
        assert data["env"]["DATABASE_URL"] == "postgres://x"


def test_parse_notebook_cleans_up_stale_empty_env_block(tmp_path: Path):
    """Opening a notebook with a legacy sensitive-only env block rewrites it.

    Covers the migration path for notebooks checked in with noise from
    an earlier Runtime-panel interaction.
    """
    notebook_dir = create_notebook(tmp_path, "Stale Env Cleanup")
    # Simulate the pre-fix state: an empty [env] block with a blanked
    # sensitive-key placeholder.
    notebook_toml = notebook_dir / "notebook.toml"
    with open(notebook_toml, "a", encoding="utf-8") as f:
        f.write('\n[env]\nOPENAI_API_KEY = ""\n')

    parse_notebook(notebook_dir)

    with open(notebook_toml, "rb") as f:
        data = tomllib.load(f)
    assert "env" not in data


def test_sensitive_env_values_stripped_on_write():
    """API keys, tokens, and passwords should not be persisted to disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Secrets Test")

        update_notebook_env(
            notebook_dir,
            {
                "OPENAI_API_KEY": "sk-proj-secret123",
                "ANTHROPIC_API_KEY": "sk-ant-secret456",
                "MY_SECRET": "hunter2",
                "AUTH_TOKEN": "tok_abc",
                "DB_PASSWORD": "p@ssw0rd",
                "DATABASE_URL": "postgres://localhost/db",
                "DEBUG": "true",
            },
        )

        with open(notebook_dir / "notebook.toml", "rb") as f:
            data = tomllib.load(f)

        env = data["env"]
        # Sensitive values stripped to empty string
        assert env["OPENAI_API_KEY"] == ""
        assert env["ANTHROPIC_API_KEY"] == ""
        assert env["MY_SECRET"] == ""
        assert env["AUTH_TOKEN"] == ""
        assert env["DB_PASSWORD"] == ""
        # Non-sensitive values preserved
        assert env["DATABASE_URL"] == "postgres://localhost/db"
        assert env["DEBUG"] == "true"


def test_create_notebook_preserves_existing_id():
    """Re-creating at the same path must keep the original notebook_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nb_dir = create_notebook(Path(tmpdir), "Stable ID")
        add_cell_to_notebook(nb_dir, "c1")
        write_cell(nb_dir, "c1", "x = 1")

        original = parse_notebook(nb_dir)
        original_id = original.id
        assert len(original.cells) == 1

        # Re-create at the same path (simulates boot() calling create again)
        nb_dir_2 = create_notebook(Path(tmpdir), "Stable ID")
        assert nb_dir_2 == nb_dir

        reopened = parse_notebook(nb_dir)
        assert reopened.id == original_id, "create_notebook must preserve the existing notebook_id"
        assert len(reopened.cells) == 1
        assert reopened.cells[0].id == "c1"


def test_update_notebook_connections_round_trip():
    """``update_notebook_connections`` writes a [connections.<name>]
    block that survives a parser round-trip. SQLite path stays
    relative on disk; resolution against the notebook dir happens
    on read."""
    from strata.notebook.models import ConnectionSpec
    from strata.notebook.parser import parse_notebook

    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "ConnTest")

        update_notebook_connections(
            notebook_dir,
            [
                ConnectionSpec(name="warehouse", driver="sqlite", path="data/db.sqlite"),
                ConnectionSpec(
                    name="prod",
                    driver="postgresql",
                    uri="postgresql://localhost:5432/prod",
                    auth={"user": "${PGUSER}", "password": "${PGPASS}"},
                ),
            ],
        )

        # Re-read the notebook and confirm both connections are back.
        state = parse_notebook(notebook_dir)
        names = {c.name for c in state.connections}
        assert names == {"warehouse", "prod"}

        warehouse = next(c for c in state.connections if c.name == "warehouse")
        # The on-disk path round-trips byte-for-byte; relative
        # paths stay relative. The cell executor resolves against
        # the notebook dir at adapter-open time so notebook.toml
        # is portable across machines.
        assert warehouse.path == "data/db.sqlite"

        prod = next(c for c in state.connections if c.name == "prod")
        assert prod.uri == "postgresql://localhost:5432/prod"
        assert prod.auth == {"user": "${PGUSER}", "password": "${PGPASS}"}


def test_update_notebook_connections_blanks_literal_secrets():
    """A literal password is scrubbed at write time. The on-disk
    body keeps the key (so the user knows which slot is configured)
    but the value is blanked."""
    from strata.notebook.models import ConnectionSpec
    from strata.notebook.parser import parse_notebook

    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "ConnSecretTest")
        update_notebook_connections(
            notebook_dir,
            [
                ConnectionSpec(
                    name="db",
                    driver="postgresql",
                    uri="postgresql://localhost/db",
                    auth={"user": "${PGUSER}", "password": "hunter2"},
                ),
            ],
        )

        state = parse_notebook(notebook_dir)
        db = next(c for c in state.connections if c.name == "db")
        # ${PGUSER} round-trips. "hunter2" is blanked.
        assert db.auth["user"] == "${PGUSER}"
        assert db.auth["password"] == ""


def test_set_variant_active_appends_entry():
    """First call appends a [[variant_group]] entry to notebook.toml."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Variants")
        set_variant_active(notebook_dir, "model", "gpt4")

        with open(notebook_dir / "notebook.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["variant_group"] == [{"group": "model", "active": "gpt4"}]


def test_set_variant_active_updates_existing_entry():
    """Second call to same group updates the existing entry in place."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Variants")
        set_variant_active(notebook_dir, "model", "gpt4")
        set_variant_active(notebook_dir, "model", "claude")

        with open(notebook_dir / "notebook.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["variant_group"] == [{"group": "model", "active": "claude"}]


def test_set_variant_active_no_op_when_unchanged():
    """Repeated identical writes don't bump updated_at."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Variants")
        set_variant_active(notebook_dir, "model", "gpt4")
        with open(notebook_dir / "notebook.toml", "rb") as f:
            first = tomllib.load(f)["updated_at"]

        set_variant_active(notebook_dir, "model", "gpt4")
        with open(notebook_dir / "notebook.toml", "rb") as f:
            second = tomllib.load(f)["updated_at"]
        assert first == second


def test_remove_variant_group_entry_drops_block_when_empty():
    """Removing the last group deletes the [[variant_group]] table entirely."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Variants")
        set_variant_active(notebook_dir, "model", "gpt4")
        remove_variant_group_entry(notebook_dir, "model")

        with open(notebook_dir / "notebook.toml", "rb") as f:
            data = tomllib.load(f)
        assert "variant_group" not in data


def test_set_variant_active_round_trips_through_parse():
    """Parser surfaces variant_active_selections from notebook.toml."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Variants")
        set_variant_active(notebook_dir, "model", "claude")
        state = parse_notebook(notebook_dir)
        assert state.variant_active_selections == {"model": "claude"}


def test_update_notebook_connections_empty_drops_block():
    """Sending an empty list deletes the [connections] table from
    notebook.toml entirely so the file doesn't carry a stub."""
    from strata.notebook.models import ConnectionSpec
    from strata.notebook.parser import parse_notebook

    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "ConnEmpty")
        update_notebook_connections(
            notebook_dir,
            [ConnectionSpec(name="db", driver="sqlite", path="db.sqlite")],
        )
        # Confirm it landed.
        with open(notebook_dir / "notebook.toml", "rb") as f:
            assert "connections" in tomllib.load(f)

        # Empty list deletes the block.
        update_notebook_connections(notebook_dir, [])
        with open(notebook_dir / "notebook.toml", "rb") as f:
            data = tomllib.load(f)
        assert "connections" not in data
        state = parse_notebook(notebook_dir)
        assert state.connections == []
