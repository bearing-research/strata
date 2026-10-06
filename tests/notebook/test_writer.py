"""Tests for notebook writer."""

import builtins
import errno
import io
import os
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
    update_cell_console_output,
    update_environment_metadata,
    update_notebook_connections,
    update_notebook_env,
    update_notebook_timeout,
    update_notebook_worker,
    update_notebook_workers,
    update_requires_python,
    write_cell,
    write_cell_tests,
    write_notebook_toml,
)


def test_create_notebook():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        notebook_dir = create_notebook(tmpdir_path, "New Notebook")

        assert notebook_dir.exists()
        assert (notebook_dir / "notebook.toml").exists()
        assert (notebook_dir / "pyproject.toml").exists()
        assert (notebook_dir / "cells").exists()
        assert (notebook_dir / "cells").is_dir()


def test_create_notebook_project_mount_adds_pinned_ro_mount():
    with tempfile.TemporaryDirectory() as tmpdir:
        parent = Path(tmpdir)
        notebook_dir = create_notebook(
            parent, "Scratch", initialize_environment=False, project_mount="project"
        )
        nb = parse_notebook(notebook_dir)
        assert len(nb.mounts) == 1
        mount = nb.mounts[0]
        assert mount.name == "project"
        # Points at the project (parent) dir, read-only, and pinned so it never
        # gets fingerprinted (no directory hashing, no staleness churn).
        assert mount.uri == f"file://{parent}"
        assert mount.mode.value == "ro"
        assert mount.pin is not None


def test_create_notebook_without_project_mount_has_no_mounts():
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Plain", initialize_environment=False)
        assert parse_notebook(notebook_dir).mounts == []


def test_create_notebook_project_mount_rejects_bad_identifier():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="identifier"):
            create_notebook(
                Path(tmpdir), "Bad", initialize_environment=False, project_mount="not-an-ident"
            )


def test_update_environment_metadata_reads_pyvenv_cfg_without_subprocess(
    monkeypatch: pytest.MonkeyPatch,
):
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

        environment = load_runtime_state(notebook_dir).environment
        assert environment.runtime_python_version == "3.13.3"


def test_write_cell():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        notebook_dir = create_notebook(tmpdir_path, "Cell Write Test")

        cell_id = "test-cell"
        add_cell_to_notebook(notebook_dir, cell_id)

        source = "x = 1 + 1\ny = x * 2"
        write_cell(notebook_dir, cell_id, source)

        cells_dir = notebook_dir / "cells"
        cell_file = cells_dir / f"{cell_id}.py"
        assert cell_file.exists()
        assert cell_file.read_text() == source


def test_write_cell_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        notebook_dir = create_notebook(tmpdir_path, "Cell Write Test")

        with pytest.raises(FileNotFoundError, match="Cell .* not found"):
            write_cell(notebook_dir, "nonexistent", "code")


def test_add_cell():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        notebook_dir = create_notebook(tmpdir_path, "Add Cell Test")

        cell1_id = "cell-1"
        add_cell_to_notebook(notebook_dir, cell1_id)

        cell2_id = "cell-2"
        add_cell_to_notebook(notebook_dir, cell2_id)

        notebook_state = parse_notebook(notebook_dir)
        assert len(notebook_state.cells) == 2
        assert notebook_state.cells[0].id == cell1_id
        assert notebook_state.cells[1].id == cell2_id


def test_add_cell_picks_extension_by_language():
    """Languages with a dedicated extension get it; others default to ``.py``.

    Tools opening cell files outside the UI (R-aware editors, linters, grep) rely on it.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Extension Test")
        cells_dir = notebook_dir / "cells"

        cases = [
            ("py-cell", "python", "py"),
            ("md-cell", "markdown", "md"),
            ("sql-cell", "sql", "py"),  # SQL cells keep the .py extension.
            ("prompt-cell", "prompt", "py"),
            ("r-cell", "r", "r"),
            ("widget-cell", "widget", "widget"),
        ]
        for cell_id, language, expected_ext in cases:
            add_cell_to_notebook(notebook_dir, cell_id, language=language)
            assert (cells_dir / f"{cell_id}.{expected_ext}").exists(), (
                f"language={language!r} should write a .{expected_ext} file"
            )

        # Notebook.toml round-trips the language correctly for each cell.
        notebook_state = parse_notebook(notebook_dir)
        by_id = {cell.id: cell.language for cell in notebook_state.cells}
        assert by_id["r-cell"].value == "r"
        assert by_id["md-cell"].value == "markdown"
        assert by_id["widget-cell"].value == "widget"
        # Widget cells seed a starter control so they're usable immediately.
        widget_source = next(c.source for c in notebook_state.cells if c.id == "widget-cell")
        assert "slider(" in widget_source


def test_add_cell_after():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        notebook_dir = create_notebook(tmpdir_path, "Add Cell After Test")

        cell1_id = "cell-1"
        add_cell_to_notebook(notebook_dir, cell1_id)

        cell2_id = "cell-2"
        add_cell_to_notebook(notebook_dir, cell2_id)

        cell1_5_id = "cell-1.5"
        add_cell_to_notebook(notebook_dir, cell1_5_id, after_cell_id=cell1_id)

        notebook_state = parse_notebook(notebook_dir)
        assert len(notebook_state.cells) == 3
        cell_ids = [c.id for c in notebook_state.cells]
        assert cell_ids.index(cell1_id) < cell_ids.index(cell1_5_id)
        assert cell_ids.index(cell1_5_id) < cell_ids.index(cell2_id)


def test_remove_cell():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        notebook_dir = create_notebook(tmpdir_path, "Remove Cell Test")

        cell1_id = "cell-1"
        add_cell_to_notebook(notebook_dir, cell1_id)

        cell2_id = "cell-2"
        add_cell_to_notebook(notebook_dir, cell2_id)

        remove_cell_from_notebook(notebook_dir, cell1_id)

        notebook_state = parse_notebook(notebook_dir)
        assert len(notebook_state.cells) == 1
        assert notebook_state.cells[0].id == cell2_id


def test_remove_cell_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        notebook_dir = create_notebook(tmpdir_path, "Remove Cell Test")

        with pytest.raises(FileNotFoundError, match="Cell .* not found"):
            remove_cell_from_notebook(notebook_dir, "nonexistent")


def test_reorder_cells():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        notebook_dir = create_notebook(tmpdir_path, "Reorder Test")

        cell1_id = "cell-1"
        add_cell_to_notebook(notebook_dir, cell1_id)

        cell2_id = "cell-2"
        add_cell_to_notebook(notebook_dir, cell2_id)

        cell3_id = "cell-3"
        add_cell_to_notebook(notebook_dir, cell3_id)

        reorder_cells(notebook_dir, [cell2_id, cell3_id, cell1_id])

        notebook_state = parse_notebook(notebook_dir)
        cell_ids = [c.id for c in notebook_state.cells]
        assert cell_ids == [cell2_id, cell3_id, cell1_id]


def test_rename_notebook():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        notebook_dir = create_notebook(tmpdir_path, "Original Name")

        rename_notebook(notebook_dir, "New Name")

        notebook_state = parse_notebook(notebook_dir)
        assert notebook_state.name == "New Name"


def test_write_notebook_toml():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        notebook_dir = create_notebook(tmpdir_path, "TOML Test")

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

        write_notebook_toml(notebook_dir, notebook_toml)

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
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Worker Notebook")
        update_notebook_worker(notebook_dir, "gpu-default")

        notebook_state = parse_notebook(notebook_dir)
        assert notebook_state.worker == "gpu-default"


def test_update_notebook_workers():
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
        assert notebook_state.workers[1].config.url == "https://executor.internal/gpu-a100"


def test_update_notebook_timeout_and_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Notebook Runtime")
        update_notebook_timeout(notebook_dir, 7.5)
        update_notebook_env(notebook_dir, {"DATABASE_URL": "postgres://localhost/db"})

        notebook_state = parse_notebook(notebook_dir)
        assert notebook_state.timeout == 7.5
        assert notebook_state.env == {"DATABASE_URL": "postgres://localhost/db"}


def test_update_notebook_env_preserves_ai_config():
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
    """An env block of only blanked sensitive keys is skipped; it is noise in shared notebooks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Sensitive Only Test")

        update_notebook_env(notebook_dir, {"OPENAI_API_KEY": "sk-proj-secret"})

        with open(notebook_dir / "notebook.toml", "rb") as f:
            data = tomllib.load(f)

        assert "env" not in data


def test_update_notebook_writers_are_no_op_when_value_unchanged():
    """Each update_notebook_* writer bumps updated_at only on a real change; a redundant call leaves
    the file byte-identical.
    """
    from strata.notebook.writer import (
        rename_notebook,
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
    """Typing an API key must not churn notebook.toml, or examples get diffs for invisible edits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "No Churn Test")
        notebook_toml = notebook_dir / "notebook.toml"

        before = notebook_toml.read_bytes()
        update_notebook_env(notebook_dir, {"OPENAI_API_KEY": "sk-proj-secret"})
        assert notebook_toml.read_bytes() == before

        # A different sensitive-only value is also a no-op: the persisted shape is
        # identical.
        update_notebook_env(notebook_dir, {"ANTHROPIC_API_KEY": "sk-ant-other"})
        assert notebook_toml.read_bytes() == before


def test_writer_emits_native_toml_datetime_and_array_of_tables(tmp_path: Path):
    """notebook.toml writes use native TOML shapes, not stringified ones.

    ``updated_at`` stays a TOML datetime (not an ``.isoformat()`` string), and
    ``[[variant_group]]``, ``workers`` and ``mounts`` stay arrays of tables, not the inline form
    ``tomli_w`` picks for simple list-of-dicts. Either regression makes noisy diffs on every edit.
    """
    from strata.notebook.writer import (
        update_notebook_mounts,
        update_notebook_workers,
    )

    notebook_dir = create_notebook(tmp_path, "Shape Test")
    notebook_toml = notebook_dir / "notebook.toml"

    update_notebook_mounts(
        notebook_dir,
        [MountSpec(name="data", uri="s3://bucket/prefix", mode=MountMode.READ_ONLY)],
    )
    update_notebook_workers(
        notebook_dir,
        [WorkerSpec(name="local", backend=WorkerBackendType.LOCAL)],
    )
    set_variant_active(notebook_dir, "classifier", "logreg")

    text = notebook_toml.read_text(encoding="utf-8")

    # Native TOML datetime: no quotes, no T separator
    assert 'updated_at = "' not in text, (
        f"updated_at must serialize as native TOML datetime, got: {text!r}"
    )
    import re

    assert re.search(
        r"^updated_at = \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
        text,
        re.MULTILINE,
    ), f"expected `updated_at = YYYY-MM-DD HH:MM:SS...`, got: {text!r}"

    # Array-of-tables, not inline-table-in-array
    assert "[[variant_group]]" in text, text
    assert "variant_group = [" not in text, text
    assert "[[mounts]]" in text, text
    assert "mounts = [" not in text, text
    assert "[[workers]]" in text, text
    assert "workers = [" not in text, text

    # Round-trips cleanly via tomllib (parser-level variant resolution
    # requires actual cells which this test deliberately omits).
    with open(notebook_toml, "rb") as f:
        data = tomllib.load(f)
    assert data["variant_group"] == [{"group": "classifier", "active": "logreg"}]
    assert isinstance(data["updated_at"], datetime)


def test_env_block_persists_when_mixed_with_non_sensitive():
    """A sensitive key alongside any non-sensitive value keeps the block."""
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
    """Opening a notebook with a legacy sensitive-only env block rewrites it."""
    notebook_dir = create_notebook(tmp_path, "Stale Env Cleanup")
    # An empty [env] block holding only a blanked sensitive-key placeholder.
    notebook_toml = notebook_dir / "notebook.toml"
    with open(notebook_toml, "a", encoding="utf-8") as f:
        f.write('\n[env]\nOPENAI_API_KEY = ""\n')

    parse_notebook(notebook_dir)

    with open(notebook_toml, "rb") as f:
        data = tomllib.load(f)
    assert "env" not in data


def test_sensitive_env_values_stripped_on_write():
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
        assert env["OPENAI_API_KEY"] == ""
        assert env["ANTHROPIC_API_KEY"] == ""
        assert env["MY_SECRET"] == ""
        assert env["AUTH_TOKEN"] == ""
        assert env["DB_PASSWORD"] == ""
        assert env["DATABASE_URL"] == "postgres://localhost/db"
        assert env["DEBUG"] == "true"


def test_create_notebook_preserves_existing_id():
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


def test_create_notebook_leaves_an_existing_notebook_as_it_is():
    """``strata new`` on an existing notebook must keep its env, workers, mounts, connections and
    dependencies, not just its id and cells.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        nb_dir = create_notebook(Path(tmpdir), "Configured", initialize_environment=False)
        add_cell_to_notebook(nb_dir, "c1")
        write_cell(nb_dir, "c1", "x = 1")
        toml_path = nb_dir / "notebook.toml"
        toml_path.write_text(
            toml_path.read_text() + '\n[env]\nREGION = "eu"\n\n[ai]\nmodel = "claude-sonnet-5"\n'
        )
        pyproject = nb_dir / "pyproject.toml"
        pyproject.write_text(
            pyproject.read_text().replace("dependencies = [", 'dependencies = [\n    "polars",')
        )
        before = (toml_path.read_bytes(), pyproject.read_bytes())

        assert create_notebook(Path(tmpdir), "Configured", initialize_environment=False) == nb_dir

        assert (toml_path.read_bytes(), pyproject.read_bytes()) == before


def test_update_notebook_connections_round_trip():
    """SQLite paths stay relative on disk; they resolve against the notebook dir on read."""
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

        state = parse_notebook(notebook_dir)
        names = {c.name for c in state.connections}
        assert names == {"warehouse", "prod"}

        warehouse = next(c for c in state.connections if c.name == "warehouse")
        # Relative paths round-trip unchanged; the cell executor resolves them against the
        # notebook dir at open time, so notebook.toml stays portable across machines.
        assert warehouse.path == "data/db.sqlite"

        prod = next(c for c in state.connections if c.name == "prod")
        assert prod.uri == "postgresql://localhost:5432/prod"
        assert prod.auth == {"user": "${PGUSER}", "password": "${PGPASS}"}


def test_update_notebook_connections_blanks_literal_secrets():
    """The key stays (showing which slot is configured) but the literal value is blanked."""
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
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Variants")
        set_variant_active(notebook_dir, "model", "gpt4")

        with open(notebook_dir / "notebook.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["variant_group"] == [{"group": "model", "active": "gpt4"}]


def test_set_variant_active_updates_existing_entry():
    """A second call for the same group updates its entry in place."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Variants")
        set_variant_active(notebook_dir, "model", "gpt4")
        set_variant_active(notebook_dir, "model", "claude")

        with open(notebook_dir / "notebook.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["variant_group"] == [{"group": "model", "active": "claude"}]


def test_set_variant_active_no_op_when_unchanged():
    """Repeated identical writes do not bump updated_at."""
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
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Variants")
        set_variant_active(notebook_dir, "model", "claude")
        state = parse_notebook(notebook_dir)
        assert state.variant_active_selections == {"model": "claude"}


def test_update_notebook_connections_empty_drops_block():
    """An empty list deletes the [connections] table rather than leaving a stub."""
    from strata.notebook.models import ConnectionSpec
    from strata.notebook.parser import parse_notebook

    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "ConnEmpty")
        update_notebook_connections(
            notebook_dir,
            [ConnectionSpec(name="db", driver="sqlite", path="db.sqlite")],
        )
        with open(notebook_dir / "notebook.toml", "rb") as f:
            assert "connections" in tomllib.load(f)

        # Empty list deletes the block.
        update_notebook_connections(notebook_dir, [])
        with open(notebook_dir / "notebook.toml", "rb") as f:
            data = tomllib.load(f)
        assert "connections" not in data
        state = parse_notebook(notebook_dir)
        assert state.connections == []


def test_update_notebook_connections_empty_on_empty_is_noop():
    """Otherwise ``updated_at`` churns and arrays of tables reserialize inline on every UI save."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "ConnNoopEmpty")
        notebook_toml = notebook_dir / "notebook.toml"
        before_mtime = notebook_toml.stat().st_mtime_ns
        before_text = notebook_toml.read_text(encoding="utf-8")

        update_notebook_connections(notebook_dir, [])

        assert notebook_toml.stat().st_mtime_ns == before_mtime
        assert notebook_toml.read_text(encoding="utf-8") == before_text


class TestUpdateRequiresPython:
    def test_rewrites_to_new_minor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nb = create_notebook(
                Path(tmpdir),
                "py_update",
                python_version="3.12",
                initialize_environment=False,
            )
            update_requires_python(nb, "3.13")
            content = (nb / "pyproject.toml").read_text()
            assert 'requires-python = "==3.13.*"' in content
            assert "==3.12" not in content

    def test_returns_previous_spec_for_rollback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nb = create_notebook(
                Path(tmpdir),
                "py_rollback",
                python_version="3.12",
                initialize_environment=False,
            )
            old = update_requires_python(nb, "3.13")
            # The previous spec is the canonical format the writer produced.
            assert old == "==3.12.*"

    def test_no_change_when_minor_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nb = create_notebook(
                Path(tmpdir),
                "py_noop",
                python_version="3.12",
                initialize_environment=False,
            )
            mtime_before = (nb / "pyproject.toml").stat().st_mtime_ns
            old = update_requires_python(nb, "3.12")
            assert old == "==3.12.*"
            # File should not be rewritten in the no-op case.
            assert (nb / "pyproject.toml").stat().st_mtime_ns == mtime_before

    def test_missing_pyproject_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            update_requires_python(tmp_path, "3.12")

    def test_legacy_range_spec_is_rewritten_to_canonical(self):
        """A legacy range spec is rewritten to the canonical ``==X.Y.*`` form on update."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nb = create_notebook(
                Path(tmpdir),
                "legacy_range",
                python_version="3.12",
                initialize_environment=False,
            )
            pyproject = nb / "pyproject.toml"
            content = pyproject.read_text()
            pyproject.write_text(
                content.replace('requires-python = "==3.12.*"', 'requires-python = ">=3.12,<3.13"')
            )
            old = update_requires_python(nb, "3.13")
            assert old == ">=3.12,<3.13"
            assert 'requires-python = "==3.13.*"' in pyproject.read_text()


def test_worker_config_model_round_trips_through_toml(tmp_path):
    """Typed known keys and extras survive a TOML round-trip; an empty config is omitted."""
    from strata.notebook.models import WorkerBackendType, WorkerConfig, WorkerSpec
    from strata.notebook.parser import parse_notebook
    from strata.notebook.writer import update_notebook_workers

    nb = tmp_path / "nb"
    nb.mkdir()
    (nb / "notebook.toml").write_text('id = "x"\nname = "x"\n')

    update_notebook_workers(
        nb,
        [
            WorkerSpec(name="local", backend=WorkerBackendType.LOCAL),  # empty config
            WorkerSpec(
                name="gpu",
                backend=WorkerBackendType.EXECUTOR,
                config=WorkerConfig(url="http://exec", transport="direct", region="us"),
            ),
        ],
    )

    state = parse_notebook(nb)
    by_name = {w.name: w for w in state.workers}
    assert isinstance(by_name["gpu"].config, WorkerConfig)
    assert by_name["gpu"].config.url == "http://exec"
    assert by_name["gpu"].config.transport == "direct"
    assert by_name["gpu"].config.model_extra.get("region") == "us"  # extra preserved
    # empty config serializes to nothing meaningful
    assert by_name["local"].config.url is None


class TestAtomicNotebookTomlWrites:
    """A crash or full disk mid-dump must leave the previous complete notebook.toml, never an empty
    or torn one.
    """

    def test_failed_dump_leaves_previous_file_intact(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            notebook_dir = create_notebook(Path(tmpdir), "Atomic")
            toml_path = notebook_dir / "notebook.toml"
            before = toml_path.read_bytes()
            assert before  # sanity

            def torn_dump(data, fp):
                fp.write(b"[partial")  # simulate a mid-write crash
                raise OSError("disk full")

            monkeypatch.setattr(writer_module, "_dump_notebook_toml", torn_dump)
            with pytest.raises(OSError, match="disk full"):
                writer_module.update_notebook_timeout(notebook_dir, 9.0)

            # The original file is untouched and no temp litter remains.
            assert toml_path.read_bytes() == before
            assert not list(notebook_dir.glob(".notebook.toml.*"))

    def test_successful_write_replaces_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            notebook_dir = create_notebook(Path(tmpdir), "Atomic OK")
            update_notebook_timeout(notebook_dir, 4.0)
            assert parse_notebook(notebook_dir).timeout == 4.0
            assert not list(notebook_dir.glob(".notebook.toml.*"))


def test_update_notebook_env_keeps_writer_conventions():
    """Like every notebook.toml writer: native TOML datetime for updated_at and [[workers]] kept as
    an array of tables.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Env Conventions")
        update_notebook_workers(
            notebook_dir,
            [
                WorkerSpec(
                    name="gpu",
                    backend=WorkerBackendType.EXECUTOR,
                    config={"url": "http://localhost:9000"},
                )
            ],
        )

        update_notebook_env(notebook_dir, {"DATABASE_URL": "postgres://localhost/db"})

        raw = (notebook_dir / "notebook.toml").read_text(encoding="utf-8")
        assert "[[workers]]" in raw  # array-of-tables survived the env save
        with open(notebook_dir / "notebook.toml", "rb") as f:
            data = tomllib.load(f)
        assert isinstance(data["updated_at"], datetime)  # not an ISO string


def test_reorder_cells_keeps_cells_the_caller_never_saw():
    """Reordering must never delete a cell.

    Callers pass a snapshot from when they opened the notebook, so a cell added since (by a server
    session, the TUI or another CLI process) would be dropped from committed config.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Concurrent Reorder")
        for cell_id in ("cell-1", "cell-2", "cell-3"):
            add_cell_to_notebook(notebook_dir, cell_id)

        # A cell the reordering caller has never heard of.
        add_cell_to_notebook(notebook_dir, "added-elsewhere")

        reorder_cells(notebook_dir, ["cell-3", "cell-1", "cell-2"])

        cell_ids = [c.id for c in parse_notebook(notebook_dir).cells]
        assert "added-elsewhere" in cell_ids
        # The requested order still holds for the cells that were named.
        assert cell_ids[:3] == ["cell-3", "cell-1", "cell-2"]


def test_reorder_cells_renumbers_every_cell_contiguously():
    """Preserved cells need an order too, or they sort unpredictably."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "Renumber")
        for cell_id in ("a", "b", "c"):
            add_cell_to_notebook(notebook_dir, cell_id)

        reorder_cells(notebook_dir, ["c", "a"])

        with open(notebook_dir / "notebook.toml", "rb") as handle:
            orders = [cell["order"] for cell in tomllib.load(handle)["cells"]]
        assert orders == list(range(len(orders)))


class _FullDiskFile:
    """A file handle whose writes fail as on a full disk, after the open succeeded."""

    def __init__(self, handle):
        self._handle = handle

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._handle.close()

    def write(self, data):
        raise OSError(errno.ENOSPC, "No space left on device")

    def __getattr__(self, name):
        return getattr(self._handle, name)


def _fill_disk_under(root: Path, monkeypatch) -> None:
    """Make every write-mode open under *root* fail its writes with ENOSPC."""
    real_open = builtins.open

    def full_disk_open(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if (
            isinstance(file, (str, os.PathLike))
            and any(flag in mode for flag in "wxa")
            and Path(file).resolve().is_relative_to(root.resolve())
        ):
            return _FullDiskFile(handle)
        return handle

    monkeypatch.setattr(builtins, "open", full_disk_open)
    # Path.write_text opens through io.open.
    monkeypatch.setattr(io, "open", full_disk_open)


@pytest.mark.parametrize(
    ("relative", "rewrite"),
    [
        ("cells/c1.py", lambda nb: write_cell(nb, "c1", "x = 2")),
        ("cells/c1.test.py", lambda nb: write_cell_tests(nb, "c1", "def test_b(): pass")),
        ("pyproject.toml", lambda nb: update_requires_python(nb, "3.13")),
        (".strata/console/c1.json", lambda nb: update_cell_console_output(nb, "c1", "new", "")),
    ],
)
def test_a_full_disk_mid_write_keeps_the_last_good_file(tmp_path, monkeypatch, relative, rewrite):
    nb = create_notebook(tmp_path, "Full Disk", python_version="3.12", initialize_environment=False)
    add_cell_to_notebook(nb, "c1")
    write_cell(nb, "c1", "x = 1")
    write_cell_tests(nb, "c1", "def test_a(): pass")
    update_cell_console_output(nb, "c1", "old", "")
    target = nb / relative
    before = target.read_bytes()
    assert before

    _fill_disk_under(nb, monkeypatch)
    with pytest.raises(OSError) as excinfo:
        rewrite(nb)
    monkeypatch.undo()

    assert excinfo.value.errno == errno.ENOSPC
    assert target.read_bytes() == before
    assert not [p for p in target.parent.iterdir() if p.name.endswith(".tmp")]


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_a_rewrite_keeps_a_mode_the_user_narrowed(tmp_path):
    nb = create_notebook(tmp_path, "Private", python_version="3.12", initialize_environment=False)
    toml_path = nb / "notebook.toml"
    os.chmod(toml_path, 0o600)

    update_notebook_timeout(nb, 12.0)

    assert toml_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "edit",
    [
        lambda nb: add_cell_to_notebook(nb, "c2"),
        lambda nb: rename_notebook(nb, "Renamed"),
        lambda nb: update_notebook_timeout(nb, 30.0),
        lambda nb: remove_cell_from_notebook(nb, "c1"),
    ],
)
def test_a_structural_edit_drops_a_leftover_owner_key(tmp_path, edit):
    """0.8.0 notebooks may carry ``owner``; nothing reads it any more."""
    nb = create_notebook(tmp_path, "Owned", initialize_environment=False)
    add_cell_to_notebook(nb, "c1")
    toml_path = nb / "notebook.toml"
    toml_path.write_text('owner = "alice@example.com"\n' + toml_path.read_text())
    assert "owner" in tomllib.loads(toml_path.read_text())

    edit(nb)

    assert "owner" not in tomllib.loads(toml_path.read_text())


def test_a_none_inside_a_display_value_stays_null_in_runtime_json(tmp_path):
    """runtime.json is JSON; turning None into "" shows the wrong value in exports."""
    import json

    from strata.notebook.writer import update_cell_display_outputs

    nb = create_notebook(tmp_path, "Nulls", initialize_environment=False)
    add_cell_to_notebook(nb, "c1")
    preview = {"mean": 2.5, "greeting": None, "rows": [1, None]}

    update_cell_display_outputs(nb, "c1", [{"content_type": "json/object", "preview": preview}])

    saved = json.loads((nb / ".strata" / "runtime.json").read_text())["cells"]["c1"]
    assert saved["display_outputs"][0]["preview"] == preview
    assert saved["display"]["preview"] == preview
