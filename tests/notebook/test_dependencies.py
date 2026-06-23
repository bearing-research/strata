"""Tests for dependency management: add, remove, list, REST + WS endpoints.

Validates:
- dependencies.py core operations (list, add, remove)
- REST endpoints (GET/POST/DELETE /v1/notebooks/{id}/dependencies)
- WebSocket messages (dependency_add, dependency_remove → dependency_changed)
- Lockfile hash change detection
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from strata.notebook.dependencies import (
    DependencyChangeResult,
    EnvironmentOperationLog,
    RPackageInfo,
    RPackageListing,
    add_dependency,
    ensure_dev_tool,
    export_requirements_text,
    import_environment_yaml_text,
    import_environment_yaml_text_streaming,
    import_requirements_text,
    import_requirements_text_streaming,
    is_valid_r_package_name,
    list_dependencies,
    list_r_packages,
    list_resolved_dependencies,
    parse_environment_yaml_text,
    parse_requirements_text,
    preview_environment_yaml_text,
    preview_requirements_text,
    remove_dependency,
    renv_add,
    renv_init,
)
from strata.notebook.executor import CellExecutor
from strata.notebook.models import CellStaleness, CellStatus
from strata.notebook.session import DependencyMutationOutcome
from strata.notebook.writer import create_notebook
from tests.notebook.e2e_fixtures import (
    NotebookBuilder,
    create_test_app,
    open_notebook_session,
    ws_connect,
)

pytestmark = pytest.mark.integration

# ============================================================================
# Core dependency operations
# ============================================================================


class TestListDependencies:
    """list_dependencies() parses pyproject.toml."""

    def test_empty_notebook(self, tmp_path: Path):
        """Newly created notebook ships the notebook-runtime baseline deps.

        See writer.create_notebook — every generated pyproject.toml
        pins pyarrow, orjson, and cloudpickle. The runtime (harness,
        pool_worker, serializer) imports all three unconditionally.
        """
        nb_dir = create_notebook(tmp_path, "empty")
        deps = list_dependencies(nb_dir)
        names = sorted(d.name for d in deps)
        assert names == ["cloudpickle", "orjson", "pyarrow"]

    def test_after_add(self, tmp_path: Path):
        """After adding a dep, it appears in the list."""
        nb_dir = create_notebook(tmp_path, "with_dep")
        result = add_dependency(nb_dir, "six")
        assert result.success
        deps = list_dependencies(nb_dir)
        names = [d.name for d in deps]
        assert "six" in names

    def test_with_version_specifier(self, tmp_path: Path):
        """Version specifiers are parsed correctly."""
        nb_dir = create_notebook(tmp_path, "versioned")
        add_dependency(nb_dir, "six>=1.0")
        deps = list_dependencies(nb_dir)
        six_dep = next((d for d in deps if d.name == "six"), None)
        assert six_dep is not None
        assert six_dep.specifier is not None
        assert ">=" in str(six_dep.specifier)

    def test_no_pyproject(self, tmp_path: Path):
        """No pyproject.toml → empty list."""
        deps = list_dependencies(tmp_path)
        assert deps == []


class TestAddDependency:
    """add_dependency() calls uv add."""

    def test_add_package(self, tmp_path: Path):
        """Adding a real package succeeds."""
        nb_dir = create_notebook(tmp_path, "add_test")
        result = add_dependency(nb_dir, "six")
        assert result.success
        assert result.action == "add"
        assert result.package == "six"
        assert result.lockfile_changed is True

    def test_add_already_present(self, tmp_path: Path):
        """Adding an existing dependency is idempotent."""
        nb_dir = create_notebook(tmp_path, "double_add")
        add_dependency(nb_dir, "six")
        result = add_dependency(nb_dir, "six")
        # uv add is idempotent — should still succeed
        assert result.success

    def test_add_nonexistent_package(self, tmp_path: Path):
        """Adding a package that doesn't exist fails."""
        nb_dir = create_notebook(tmp_path, "bad_pkg")
        result = add_dependency(nb_dir, "this-package-definitely-does-not-exist-xyz123")
        assert result.success is False
        assert result.error is not None

    def test_dev_flag_passes_uv_add_dev(self, tmp_path: Path):
        """``dev=True`` runs ``uv add --dev`` (so the tool lands in the dev group)."""
        nb_dir = create_notebook(tmp_path, "dev_arg")
        captured: dict = {}
        original = add_dependency.__globals__["_run_uv_command"]

        def _capture(notebook_dir, args, **kwargs):
            captured["args"] = args
            return original(notebook_dir, args, **kwargs)

        with patch("strata.notebook.dependencies._run_uv_command", side_effect=_capture):
            result = add_dependency(nb_dir, "iniconfig", dev=True)

        assert result.success
        assert captured["args"] == ["add", "--dev", "iniconfig"]

    def test_dev_dependency_not_in_runtime_deps(self, tmp_path: Path):
        """A dev add lands in [dependency-groups] dev, not [project.dependencies]."""
        import tomllib

        nb_dir = create_notebook(tmp_path, "dev_group")
        assert add_dependency(nb_dir, "iniconfig", dev=True).success

        data = tomllib.loads((nb_dir / "pyproject.toml").read_text())
        runtime = data.get("project", {}).get("dependencies", [])
        dev = data.get("dependency-groups", {}).get("dev", [])
        assert not any("iniconfig" in d for d in runtime)
        assert any("iniconfig" in d for d in dev)

    def test_ensure_dev_tool_installs_into_dev_group(self, tmp_path: Path):
        """``ensure_dev_tool`` is the generic provisioning entry (dev group)."""
        import tomllib

        nb_dir = create_notebook(tmp_path, "ensure_tool")
        result = ensure_dev_tool(nb_dir, "iniconfig")

        assert result.success
        data = tomllib.loads((nb_dir / "pyproject.toml").read_text())
        dev = data.get("dependency-groups", {}).get("dev", [])
        assert any("iniconfig" in d for d in dev)

    def test_add_when_uv_missing(self, tmp_path: Path):
        """Returns failure when uv is not available."""
        with patch(
            "strata.notebook.dependencies.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            result = add_dependency(tmp_path, "requests")
            assert result.success is False
            assert result.error is not None
            assert "uv not found" in result.error

    def test_add_records_operation_log(self, tmp_path: Path):
        """Successful add operations should expose uv command details."""
        nb_dir = create_notebook(tmp_path, "add_log")
        with patch(
            "strata.notebook.dependencies.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["uv", "add", "requests"],
                returncode=0,
                stdout="Resolved 1 package\nInstalled requests\n",
                stderr="Using cached wheels\n",
            ),
        ):
            result = add_dependency(nb_dir, "requests")

        assert result.success is True
        assert result.operation_log is not None
        assert result.operation_log.command == "uv add requests"
        assert "Resolved 1 package" in result.operation_log.stdout
        assert "Using cached wheels" in result.operation_log.stderr
        assert result.operation_log.duration_ms is not None


class TestRemoveDependency:
    """remove_dependency() calls uv remove."""

    def test_remove_package(self, tmp_path: Path):
        """Removing an added package succeeds."""
        nb_dir = create_notebook(tmp_path, "remove_test")
        add_dependency(nb_dir, "six")
        result = remove_dependency(nb_dir, "six")
        assert result.success
        assert result.action == "remove"
        assert result.lockfile_changed is True

        # Verify it's gone
        deps = list_dependencies(nb_dir)
        names = [d.name for d in deps]
        assert "six" not in names

    def test_remove_nonexistent(self, tmp_path: Path):
        """Removing a package that isn't present fails."""
        nb_dir = create_notebook(tmp_path, "remove_missing")
        result = remove_dependency(nb_dir, "this-package-not-installed")
        assert result.success is False
        assert result.error is not None


class TestRequirementsCompatibility:
    """requirements.txt export/import helpers."""

    def test_export_requirements_text(self, tmp_path: Path):
        """Export should preserve direct dependency specifiers."""
        nb_dir = create_notebook(tmp_path, "requirements_export")
        add_dependency(nb_dir, "six==1.17.0")

        exported = export_requirements_text(nb_dir)

        assert exported.endswith("\n")
        assert "pyarrow>=18.0.0" in exported
        assert "six==1.17.0" in exported

    def test_import_requirements_text_replaces_direct_dependencies(self, tmp_path: Path):
        """Import should replace the notebook's direct dependency set."""
        nb_dir = create_notebook(tmp_path, "requirements_import")
        add_dependency(nb_dir, "requests")

        result = import_requirements_text(
            nb_dir,
            "pyarrow>=18.0.0\nsix==1.17.0\n",
        )

        assert result.success is True
        assert result.imported_count == 2
        names = [dep.name for dep in result.dependencies]
        assert "pyarrow" in names
        assert "six" in names
        assert "requests" not in names

    def test_import_requirements_records_operation_log(self, tmp_path: Path):
        """requirements.txt imports should preserve the underlying uv sync details."""
        nb_dir = create_notebook(tmp_path, "requirements_log")

        with patch(
            "strata.notebook.dependencies.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["uv", "sync"],
                returncode=0,
                stdout="Resolved 2 packages\n",
                stderr="Prepared environment\n",
            ),
        ):
            result = import_requirements_text(
                nb_dir,
                "pyarrow>=18.0.0\nsix==1.17.0\n",
            )

        assert result.success is True
        assert result.operation_log is not None
        assert result.operation_log.command == "uv sync"
        assert "Resolved 2 packages" in result.operation_log.stdout
        assert "Prepared environment" in result.operation_log.stderr

    @pytest.mark.asyncio
    async def test_import_requirements_text_streaming_records_live_output(
        self, tmp_path: Path, monkeypatch
    ):
        """Streaming requirements import should preserve command output and success state."""
        nb_dir = create_notebook(tmp_path, "requirements_streaming")

        async def _fake_run_uv_command_streaming(
            notebook_dir: Path,
            args: list[str],
            *,
            timeout: int,
            display_name: str,
            on_update=None,
        ):
            del notebook_dir
            del timeout
            del display_name
            if on_update is not None:
                await on_update("stdout", "Resolved 2 packages\n", False)
            return type(
                "FakeUvResult",
                (),
                {
                    "success": True,
                    "error": None,
                    "operation_log": EnvironmentOperationLog(
                        command=" ".join(["uv", *args]),
                        duration_ms=11,
                        stdout="Resolved 2 packages\n",
                        stderr="Prepared environment\n",
                        stdout_truncated=False,
                        stderr_truncated=False,
                    ),
                },
            )()

        monkeypatch.setattr(
            "strata.notebook.dependencies.run_uv_command_streaming",
            _fake_run_uv_command_streaming,
        )

        result = await import_requirements_text_streaming(
            nb_dir,
            "pyarrow>=18.0.0\nsix==1.17.0\n",
        )

        assert result.success is True
        assert result.operation_log is not None
        assert result.operation_log.command == "uv sync"
        assert "Resolved 2 packages" in result.operation_log.stdout
        assert "Prepared environment" in result.operation_log.stderr

    def test_parse_requirements_text_rejects_pip_flags(self):
        """Unsupported pip-style directives should fail clearly."""
        with pytest.raises(ValueError, match="Unsupported requirements entry"):
            parse_requirements_text("-r base.txt")

    def test_parse_environment_yaml_text_extracts_pip_compatible_requirements(self):
        """environment.yaml import should translate a supported subset with warnings."""
        requirements, warnings = parse_environment_yaml_text(
            """
name: demo
channels:
  - conda-forge
dependencies:
  - python=3.13
  - pyarrow=22.0.0
  - six=1.17.0
  - pip
  - pip:
      - requests==2.32.3
"""
        )

        assert requirements == ["pyarrow==22.0.0", "six==1.17.0", "requests==2.32.3"]
        assert any("channels" in warning for warning in warnings)
        assert any("python version pin" in warning for warning in warnings)

    def test_import_environment_yaml_text_replaces_direct_dependencies(self, tmp_path: Path):
        """environment.yaml import should best-effort replace direct dependencies."""
        nb_dir = create_notebook(tmp_path, "environment_yaml_import")
        add_dependency(nb_dir, "requests")

        result = import_environment_yaml_text(
            nb_dir,
            """
name: demo
dependencies:
  - pyarrow=22.0.0
  - six=1.17.0
  - pip:
      - urllib3==2.5.0
""",
        )

        assert result.success is True
        assert result.imported_count == 3
        names = [dep.name for dep in result.dependencies]
        assert "pyarrow" in names
        assert "six" in names
        assert "urllib3" in names
        assert "requests" not in names

    @pytest.mark.asyncio
    async def test_import_environment_yaml_text_streaming_preserves_warnings(
        self, tmp_path: Path, monkeypatch
    ):
        """Streaming environment.yaml import should keep best-effort warnings."""
        nb_dir = create_notebook(tmp_path, "environment_yaml_streaming")

        async def _fake_run_uv_command_streaming(
            notebook_dir: Path,
            args: list[str],
            *,
            timeout: int,
            display_name: str,
            on_update=None,
        ):
            del notebook_dir
            del timeout
            del display_name
            if on_update is not None:
                await on_update("stderr", "Resolving translated environment\n", False)
            return type(
                "FakeUvResult",
                (),
                {
                    "success": True,
                    "error": None,
                    "operation_log": EnvironmentOperationLog(
                        command=" ".join(["uv", *args]),
                        duration_ms=13,
                        stdout="",
                        stderr="Resolving translated environment\n",
                        stdout_truncated=False,
                        stderr_truncated=False,
                    ),
                },
            )()

        monkeypatch.setattr(
            "strata.notebook.dependencies.run_uv_command_streaming",
            _fake_run_uv_command_streaming,
        )

        result = await import_environment_yaml_text_streaming(
            nb_dir,
            """
name: demo
channels:
  - conda-forge
dependencies:
  - python=3.13
  - pyarrow=22.0.0
  - six=1.17.0
""",
        )

        assert result.success is True
        assert any("channels" in warning for warning in result.warnings)
        assert any("python version pin" in warning for warning in result.warnings)
        assert result.operation_log is not None
        assert result.operation_log.command == "uv sync"

    def test_list_resolved_dependencies_reads_uv_lock(self, tmp_path: Path):
        """Resolved dependencies should be listed from uv.lock."""
        nb_dir = create_notebook(tmp_path, "resolved_list")
        add_dependency(nb_dir, "six==1.17.0")

        resolved = list_resolved_dependencies(nb_dir)

        names = [dep.name for dep in resolved]
        assert "pyarrow" in names
        assert "six" in names
        six_dep = next(dep for dep in resolved if dep.name == "six")
        assert str(six_dep.version) == "1.17.0"

    def test_preview_requirements_text_reports_diff(self, tmp_path: Path):
        """Requirements preview should report additions, removals, and unchanged deps."""
        nb_dir = create_notebook(tmp_path, "requirements_preview")
        add_dependency(nb_dir, "requests==2.32.3")

        preview = preview_requirements_text(
            nb_dir,
            "pyarrow>=18.0.0\norjson>=3.10.0\ncloudpickle>=3.0.0\nsix==1.17.0\n",
        )

        assert preview.imported_count == 4
        assert [dep.name for dep in preview.additions] == ["six"]
        assert [dep.name for dep in preview.removals] == ["requests"]
        assert sorted(dep.name for dep in preview.unchanged) == [
            "cloudpickle",
            "orjson",
            "pyarrow",
        ]

    def test_preview_environment_yaml_text_reports_warnings_and_diff(self, tmp_path: Path):
        """environment.yaml preview should translate, warn, and diff dependencies."""
        nb_dir = create_notebook(tmp_path, "environment_yaml_preview")
        add_dependency(nb_dir, "requests==2.32.3")

        preview = preview_environment_yaml_text(
            nb_dir,
            """
name: demo
channels:
  - conda-forge
dependencies:
  - python=3.13
  - pyarrow=22.0.0
  - six=1.17.0
""",
        )

        assert preview.imported_count == 2
        assert any("channels" in warning for warning in preview.warnings)
        assert any("python version pin" in warning for warning in preview.warnings)
        # Current notebook baseline (pyarrow, orjson, cloudpickle) +
        # requests. YAML pins pyarrow at a different version and adds
        # six. Version differences produce matching remove+add entries.
        assert [dep.name for dep in preview.additions] == ["pyarrow", "six"]
        assert sorted(dep.name for dep in preview.removals) == [
            "cloudpickle",
            "orjson",
            "pyarrow",
            "requests",
        ]
        assert preview.unchanged == []


# ============================================================================
# REST API tests
# ============================================================================


class TestDependencyRESTEndpoints:
    """REST endpoints for dependency management."""

    @pytest.fixture
    def setup(self):
        app = create_test_app()
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmpdir:
            yield client, Path(tmpdir)

    def test_list_dependencies_empty(self, setup):
        """GET /dependencies on a fresh notebook returns empty list."""
        client, tmp = setup
        nb = NotebookBuilder(tmp)

        with open_notebook_session(client, nb.path) as (sid, session):
            resp = client.get(f"/v1/notebooks/{sid}/dependencies")
            assert resp.status_code == 200
            data = resp.json()
            assert "dependencies" in data
            assert isinstance(data["dependencies"], list)
            assert "resolved_dependencies" in data
            assert "environment" in data
            assert "sync_state" in data["environment"]

    def test_add_dependency_rest(self, setup):
        """POST /dependencies adds a package."""
        client, tmp = setup
        nb = NotebookBuilder(tmp)

        with open_notebook_session(client, nb.path) as (sid, session):
            resp = client.post(
                f"/v1/notebooks/{sid}/dependencies",
                json={"package": "six"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["package"] == "six"
            assert data["operation_log"]["command"] == "uv add six"
            assert "environment" in data
            assert "declared_package_count" in data["environment"]
            assert "stale_cell_count" in data

            # Verify via list
            resp2 = client.get(f"/v1/notebooks/{sid}/dependencies")
            deps = resp2.json()["dependencies"]
            names = [d["name"] for d in deps]
            assert "six" in names

    def test_add_dependency_rest_returns_updated_cells(self, setup):
        """Dependency changes return refreshed cell statuses after env invalidation."""
        client, tmp = setup
        nb = NotebookBuilder(tmp)
        nb.add_cell("c1", "x = 1")
        nb.add_cell("c2", "y = x + 1", after="c1")
        nb.add_cell("c3", "print(y)", after="c2")

        with open_notebook_session(client, nb.path) as (sid, session):

            async def _prime_cells():
                executor = CellExecutor(session)
                result1 = await executor.execute_cell("c1", "x = 1")
                result2 = await executor.execute_cell("c2", "y = x + 1")
                assert result1.success
                assert result2.success

            asyncio.run(_prime_cells())
            session.compute_staleness()
            statuses_before = {cell.id: cell.status for cell in session.notebook_state.cells}
            assert statuses_before["c1"] == CellStatus.READY
            assert statuses_before["c2"] == CellStatus.READY

            resp = client.post(
                f"/v1/notebooks/{sid}/dependencies",
                json={"package": "six"},
            )

            assert resp.status_code == 200
            data = resp.json()
            assert "cells" in data
            statuses = {cell["id"]: cell["status"] for cell in data["cells"]}
            assert statuses["c1"] == "idle"
            assert statuses["c2"] == "idle"

    def test_remove_dependency_rest(self, setup):
        """DELETE /dependencies/{package} removes a package."""
        client, tmp = setup
        nb = NotebookBuilder(tmp)

        with open_notebook_session(client, nb.path) as (sid, session):
            # Add first
            client.post(
                f"/v1/notebooks/{sid}/dependencies",
                json={"package": "six"},
            )

            # Remove
            resp = client.delete(f"/v1/notebooks/{sid}/dependencies/six")
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert "environment" in data
            assert data["operation_log"]["command"] == "uv remove six"
            assert "stale_cell_ids" in data

            # Verify removed
            resp2 = client.get(f"/v1/notebooks/{sid}/dependencies")
            deps = resp2.json()["dependencies"]
            names = [d["name"] for d in deps]
            assert "six" not in names

    def test_export_requirements_rest(self, setup):
        """GET /environment/requirements.txt exports direct dependencies."""
        client, tmp = setup
        nb = NotebookBuilder(tmp)

        with open_notebook_session(client, nb.path) as (sid, session):
            client.post(
                f"/v1/notebooks/{sid}/dependencies",
                json={"package": "six==1.17.0"},
            )

            resp = client.get(f"/v1/notebooks/{sid}/environment/requirements.txt")
            assert resp.status_code == 200
            assert "pyarrow>=18.0.0" in resp.text
            assert "six==1.17.0" in resp.text

    def test_import_requirements_rest(self, setup):
        """POST /environment/requirements.txt imports a full dependency set."""
        client, tmp = setup
        nb = NotebookBuilder(tmp)

        with open_notebook_session(client, nb.path) as (sid, session):
            resp = client.post(
                f"/v1/notebooks/{sid}/environment/requirements.txt",
                json={"requirements": "pyarrow>=18.0.0\nsix==1.17.0\n"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["imported_count"] == 2
            assert data["operation_log"]["command"] == "uv sync"
            assert "environment" in data
            assert "resolved_dependencies" in data
            names = [dep["name"] for dep in data["dependencies"]]
            assert "pyarrow" in names
            assert "six" in names

    def test_import_environment_yaml_rest(self, setup):
        """POST /environment/environment.yaml imports a supported subset with warnings."""
        client, tmp = setup
        nb = NotebookBuilder(tmp)

        with open_notebook_session(client, nb.path) as (sid, session):
            resp = client.post(
                f"/v1/notebooks/{sid}/environment/environment.yaml",
                json={
                    "environment_yaml": """
name: demo
channels:
  - conda-forge
dependencies:
  - python=3.13
  - pyarrow=22.0.0
  - six=1.17.0
  - pip:
      - requests==2.32.3
"""
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["imported_count"] == 3
            assert data["operation_log"]["command"] == "uv sync"
            assert any("channels" in warning for warning in data["warnings"])
            assert "resolved_dependencies" in data
            names = [dep["name"] for dep in data["dependencies"]]
            assert "pyarrow" in names
            assert "six" in names
            assert "requests" in names

    def test_preview_requirements_rest(self, setup):
        """POST /environment/requirements.txt/preview returns import diff."""
        client, tmp = setup
        nb = NotebookBuilder(tmp)

        with open_notebook_session(client, nb.path) as (sid, session):
            client.post(
                f"/v1/notebooks/{sid}/dependencies",
                json={"package": "requests==2.32.3"},
            )

            resp = client.post(
                f"/v1/notebooks/{sid}/environment/requirements.txt/preview",
                json={
                    "requirements": (
                        "pyarrow>=18.0.0\norjson>=3.10.0\ncloudpickle>=3.0.0\nsix==1.17.0\n"
                    )
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["imported_count"] == 4
            assert [dep["name"] for dep in data["additions"]] == ["six"]
            assert [dep["name"] for dep in data["removals"]] == ["requests"]
            assert sorted(dep["name"] for dep in data["unchanged"]) == [
                "cloudpickle",
                "orjson",
                "pyarrow",
            ]
            assert "resolved_dependencies" in data

    def test_preview_environment_yaml_rest(self, setup):
        """POST /environment/environment.yaml/preview returns warnings and import diff."""
        client, tmp = setup
        nb = NotebookBuilder(tmp)

        with open_notebook_session(client, nb.path) as (sid, session):
            client.post(
                f"/v1/notebooks/{sid}/dependencies",
                json={"package": "requests==2.32.3"},
            )

            resp = client.post(
                f"/v1/notebooks/{sid}/environment/environment.yaml/preview",
                json={
                    "environment_yaml": """
name: demo
channels:
  - conda-forge
dependencies:
  - python=3.13
  - pyarrow=22.0.0
  - six=1.17.0
"""
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["imported_count"] == 2
            assert any("channels" in warning for warning in data["warnings"])
            assert [dep["name"] for dep in data["additions"]] == ["pyarrow", "six"]
            assert sorted(dep["name"] for dep in data["removals"]) == [
                "cloudpickle",
                "orjson",
                "pyarrow",
                "requests",
            ]
            assert data["unchanged"] == []

    def test_add_bad_package_rest(self, setup):
        """POST /dependencies with invalid package returns 400."""
        client, tmp = setup
        nb = NotebookBuilder(tmp)

        with open_notebook_session(client, nb.path) as (sid, session):
            resp = client.post(
                f"/v1/notebooks/{sid}/dependencies",
                json={"package": "this-pkg-does-not-exist-xyz123"},
            )
            assert resp.status_code == 400
            detail = resp.json()["detail"]
            assert "message" in detail
            assert "operation_log" in detail
            assert detail["operation_log"]["command"] == "uv add this-pkg-does-not-exist-xyz123"

    def test_list_dependencies_404(self, setup):
        """GET /dependencies for unknown notebook returns 404."""
        client, tmp = setup
        resp = client.get("/v1/notebooks/nonexistent/dependencies")
        assert resp.status_code == 404


# ============================================================================
# WebSocket tests
# ============================================================================


class TestDependencyWebSocket:
    """WebSocket messages for dependency management."""

    @pytest.fixture
    def setup(self):
        app = create_test_app()
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmpdir:
            yield client, Path(tmpdir)

    def test_dependency_add_via_ws(self, setup):
        """dependency_add message → dependency_changed broadcast."""
        client, tmp = setup
        nb = NotebookBuilder(tmp)

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                ws.send("dependency_add", {"package": "six"})
                msg = ws.receive_until("dependency_changed")

                assert msg["payload"]["action"] == "add"
                assert msg["payload"]["package"] == "six"
                assert msg["payload"]["success"] is True
                assert msg["payload"]["lockfile_changed"] is True

    def test_dependency_remove_via_ws(self, setup):
        """dependency_remove message → dependency_changed broadcast."""
        client, tmp = setup
        nb = NotebookBuilder(tmp)

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                # Add first
                ws.send("dependency_add", {"package": "six"})
                ws.receive_until("dependency_changed")
                ws.clear()

                # Remove
                ws.send("dependency_remove", {"package": "six"})
                msg = ws.receive_until("dependency_changed")

                assert msg["payload"]["action"] == "remove"
                assert msg["payload"]["package"] == "six"
                assert msg["payload"]["success"] is True

    def test_dependency_add_missing_package(self, setup):
        """dependency_add without package field → error."""
        client, tmp = setup
        nb = NotebookBuilder(tmp)

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                ws.send("dependency_add", {})
                msg = ws.receive_until("error")
                assert "package" in msg["payload"]["error"].lower()

    def test_dependency_changed_includes_dep_list(self, setup):
        """dependency_changed includes updated dependency list."""
        client, tmp = setup
        nb = NotebookBuilder(tmp)

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                ws.send("dependency_add", {"package": "six"})
                msg = ws.receive_until("dependency_changed")

                deps = msg["payload"]["dependencies"]
                assert isinstance(deps, list)
                names = [d["name"] for d in deps]
                assert "six" in names
                assert "resolved_dependencies" in msg["payload"]
                assert "environment" in msg["payload"]
                assert "declared_package_count" in msg["payload"]["environment"]

    def test_dependency_add_via_ws_broadcasts_cell_status_updates(self, setup, monkeypatch):
        """Lockfile-changing dependency updates broadcast refreshed cell status."""
        client, tmp = setup
        nb = NotebookBuilder(tmp)
        nb.add_cell("c1", "x = 1")
        nb.add_cell("c2", "y = x + 1", after="c1")

        with open_notebook_session(client, nb.path) as (sid, session):

            async def fake_mutate_dependency(self, package, *, action):
                assert action == "add"
                result = DependencyChangeResult(
                    success=True,
                    package=package,
                    action=action,
                    lockfile_changed=True,
                    dependencies=[],
                )
                staleness_map = {
                    "c1": CellStaleness(status=CellStatus.IDLE),
                    "c2": CellStaleness(status=CellStatus.IDLE),
                }
                return DependencyMutationOutcome(
                    result=result,
                    staleness_map=staleness_map,
                )

            monkeypatch.setattr(type(session), "mutate_dependency", fake_mutate_dependency)

            with ws_connect(client, sid) as ws:
                ws.send("dependency_add", {"package": "six"})
                changed = ws.receive_until("dependency_changed")
                assert changed["payload"]["lockfile_changed"] is True
                assert changed["payload"]["package"] == "six"
                assert "cells" in changed["payload"]
                assert changed["payload"]["stale_cell_count"] == 2
                assert "environment" in changed["payload"]

                status1 = ws.receive_until("cell_status", cell_id="c1")
                status2 = ws.receive_until("cell_status", cell_id="c2")
                assert status1["payload"]["status"] == "idle"
                assert status2["payload"]["status"] == "idle"


# ============================================================================
# R package listing
# ============================================================================


class TestListRPackages:
    """``list_r_packages`` parses ``installed.packages()`` output for the UI.

    Returns an ``RPackageListing`` that explicitly distinguishes
    "the probe failed" from "the project library is empty" — the
    bare-list-returning shape lost that distinction and the UI
    rendered "no packages installed" on probe failures (Codex
    review on #88).
    """

    def test_rscript_missing_returns_status(self, monkeypatch, tmp_path):
        """No Rscript on PATH → ``status='rscript_missing'``, no spawn."""
        monkeypatch.setattr(shutil, "which", lambda name: None)
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: pytest.fail("Rscript missing → must not invoke subprocess.run"),
        )
        result = list_r_packages(tmp_path)
        assert result == RPackageListing(packages=[], status="rscript_missing", error=None)

    def test_parses_installed_packages_output(self, monkeypatch, tmp_path):
        """Two-column tab output → ``RPackageInfo`` entries, sorted by name,
        with ``status='ok'``."""
        monkeypatch.setattr(shutil, "which", lambda name: "/fake/Rscript")
        sample_stdout = "tibble\t3.2.1\narrow\t14.0.0\njsonlite\t1.8.7\n"

        def fake_run(args, **kwargs):
            assert args[0] == "/fake/Rscript"
            # Confirm we scope to the project library via renv —
            # a bare ``installed.packages()`` would enumerate every
            # ``.libPaths()`` entry (P1 from #88 review).
            assert "renv::paths$library" in args[2]
            assert "lib.loc = lib" in args[2]
            return SimpleNamespace(returncode=0, stdout=sample_stdout, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = list_r_packages(tmp_path)

        assert result.status == "ok"
        assert result.error is None
        assert result.packages == [
            RPackageInfo(name="arrow", version="14.0.0"),
            RPackageInfo(name="jsonlite", version="1.8.7"),
            RPackageInfo(name="tibble", version="3.2.1"),
        ]

    def test_renv_not_active_sentinel(self, monkeypatch, tmp_path):
        """When the R snippet can't load renv (pre-bootstrap notebook,
        broken activator) it emits the ``RENV_NOT_ACTIVE`` sentinel
        and exits 0 — surface as ``status='renv_not_active'`` so the
        UI can render a targeted hint."""
        monkeypatch.setattr(shutil, "which", lambda name: "/fake/Rscript")
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: SimpleNamespace(returncode=0, stdout="RENV_NOT_ACTIVE\n", stderr=""),
        )

        result = list_r_packages(tmp_path)

        assert result == RPackageListing(packages=[], status="renv_not_active", error=None)

    def test_empty_library_status_ok(self, monkeypatch, tmp_path):
        """Empty project library: status ``ok`` + empty packages list.
        Different from the failure modes — the UI shows "no packages
        installed" only on the ``ok``-but-empty case."""
        monkeypatch.setattr(shutil, "which", lambda name: "/fake/Rscript")
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: SimpleNamespace(returncode=0, stdout="", stderr=""),
        )

        result = list_r_packages(tmp_path)

        assert result == RPackageListing(packages=[], status="ok", error=None)

    def test_nonzero_exit_returns_failed_status(self, monkeypatch, tmp_path):
        """Rscript exits non-zero (corrupt R install, etc.) →
        ``status='failed'`` + error message from stderr. Pre-fix
        this returned the same empty list as a healthy empty
        library, and the UI couldn't tell them apart."""
        monkeypatch.setattr(shutil, "which", lambda name: "/fake/Rscript")
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: SimpleNamespace(
                returncode=1, stdout="", stderr="cannot find R installation"
            ),
        )

        result = list_r_packages(tmp_path)

        assert result.status == "failed"
        assert result.packages == []
        assert result.error == "cannot find R installation"

    def test_timeout_returns_failed_status(self, monkeypatch, tmp_path):
        """Hanging Rscript → ``status='failed'`` with a timeout message."""
        monkeypatch.setattr(shutil, "which", lambda name: "/fake/Rscript")

        def fake_run(*a, **kw):
            raise subprocess.TimeoutExpired(cmd="Rscript", timeout=30)

        monkeypatch.setattr(subprocess, "run", fake_run)

        result = list_r_packages(tmp_path)

        assert result.status == "failed"
        assert result.packages == []
        assert result.error and "timed out" in result.error

    def test_skips_malformed_lines(self, monkeypatch, tmp_path):
        """Lines without name+version columns are silently dropped — the
        loop tolerates a single bad line without losing the rest. Final
        status is still ``ok`` (the malformed lines aren't an error)."""
        monkeypatch.setattr(shutil, "which", lambda name: "/fake/Rscript")
        sample_stdout = (
            "arrow\t14.0.0\n"
            "garbage-line-no-tab\n"
            "\t\n"  # empty fields
            "tibble\t3.2.1\n"
        )
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: SimpleNamespace(returncode=0, stdout=sample_stdout, stderr=""),
        )

        result = list_r_packages(tmp_path)

        assert result.status == "ok"
        assert result.packages == [
            RPackageInfo(name="arrow", version="14.0.0"),
            RPackageInfo(name="tibble", version="3.2.1"),
        ]

    def test_runs_with_notebook_cwd(self, monkeypatch, tmp_path):
        """Rscript runs with cwd=notebook_dir so ``.Rprofile`` activates
        renv and ``renv::paths$library(project = getwd())`` resolves
        to the project's library, not the system library."""
        monkeypatch.setattr(shutil, "which", lambda name: "/fake/Rscript")
        captured: dict[str, str | None] = {}

        def fake_run(args, **kwargs):
            captured["cwd"] = kwargs.get("cwd")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        list_r_packages(tmp_path)

        assert captured["cwd"] == str(tmp_path)


# ============================================================================
# R package install + bootstrap (renv::init / renv::install)
# ============================================================================


class TestIsValidRPackageName:
    """Reject anything outside CRAN's ``[A-Za-z][A-Za-z0-9.]*`` shape.

    The name is concatenated into an Rscript ``-e`` body, so the
    validator is the only line of defence against shell or R-code
    injection. ``ggplot2`` is fine; ``ggplot2; system('rm -rf /')``
    is not.
    """

    @pytest.mark.parametrize(
        "name",
        ["arrow", "ggplot2", "data.table", "R6", "rmarkdown"],
    )
    def test_accepts_valid_names(self, name):
        assert is_valid_r_package_name(name)

    @pytest.mark.parametrize(
        "name",
        [
            "",  # empty
            "2arrow",  # starts with digit
            "ggplot-2",  # dash (not allowed by CRAN)
            "arrow; rm",  # shell metacharacter
            'arrow"; system("ls"); "',  # quote injection
            "package/with/slashes",
            "package\nname",  # newline
        ],
    )
    def test_rejects_invalid_names(self, name):
        assert not is_valid_r_package_name(name)


def _make_fake_rscript_streaming(
    *,
    success: bool,
    stdout: str = "",
    stderr: str = "",
    error: str | None = None,
    side_effect=None,
    captured_snippets: list[str] | None = None,
):
    """Build a stand-in for ``run_rscript_command_streaming``.

    Captures the snippet (so tests can assert what R code ran) and
    exercises the ``on_update`` callback (so the wiring from
    streaming → ``environment_job_progress`` is covered). Used by
    the ``renv_init`` / ``renv_add`` tests to avoid mocking
    ``asyncio.create_subprocess_exec`` line by line.
    """
    from strata.notebook.dependencies import _RscriptCommandResult

    async def fake(notebook_dir, snippet, *, timeout, display_name, on_update=None):
        del notebook_dir, timeout
        if captured_snippets is not None:
            captured_snippets.append(snippet)
        if on_update is not None:
            if stdout:
                await on_update("stdout", stdout, False)
            if stderr:
                await on_update("stderr", stderr, False)
        if side_effect is not None:
            side_effect()
        return _RscriptCommandResult(
            success=success,
            error=error or (f"{display_name} failed (exit 1)" if not success else None),
            operation_log=EnvironmentOperationLog(
                command=f"Rscript -e {snippet!r}",
                duration_ms=11,
                stdout=stdout,
                stderr=stderr,
                stdout_truncated=False,
                stderr_truncated=False,
            ),
        )

    return fake


class TestRenvInit:
    """``renv_init`` streams ``Rscript -e <bootstrap snippet>`` and
    surfaces lockfile-change + operation log to the env-panel."""

    @pytest.mark.asyncio
    async def test_rscript_missing(self, monkeypatch, tmp_path):
        """Without Rscript on PATH the streaming helper fails fast with a
        clear error before spawning a subprocess. Use the real helper
        (no mock) so we cover the actual shutil.which short-circuit."""
        monkeypatch.setattr(shutil, "which", lambda name: None)

        result = await renv_init(tmp_path)

        assert result.success is False
        assert result.action == "r_init"
        assert "Rscript not found" in (result.error or "")

    @pytest.mark.asyncio
    async def test_success_writes_lockfile_change_signal(self, monkeypatch, tmp_path):
        """A successful init creates ``renv.lock`` — we detect the new
        hash and report ``lockfile_changed=True``."""
        from strata.notebook import dependencies as deps_module

        captured: list[str] = []

        def write_lockfile():
            (tmp_path / "renv.lock").write_text('{"R": {"Version": "4.4.1"}}', encoding="utf-8")

        monkeypatch.setattr(
            deps_module,
            "run_rscript_command_streaming",
            _make_fake_rscript_streaming(
                success=True,
                stdout="renv 1.0.7 initialised\n",
                side_effect=write_lockfile,
                captured_snippets=captured,
            ),
        )

        result = await renv_init(tmp_path)

        assert result.success is True
        assert result.action == "r_init"
        assert result.lockfile_changed is True
        assert result.operation_log is not None
        snippet = captured[0]
        assert "renv::init(bare = TRUE)" in snippet
        assert 'renv::install(c("jsonlite", "arrow"))' in snippet
        assert "renv::snapshot" in snippet

    @pytest.mark.asyncio
    async def test_progress_callback_fires_during_streaming(self, monkeypatch, tmp_path):
        """The whole point of PR G: ``on_update`` must be invoked while
        the subprocess is running so ``environment_job_progress`` frames
        go out live during a multi-minute arrow compile."""
        from strata.notebook import dependencies as deps_module

        monkeypatch.setattr(
            deps_module,
            "run_rscript_command_streaming",
            _make_fake_rscript_streaming(
                success=True,
                stdout="installing arrow\n",
                stderr="compiling C++\n",
                side_effect=lambda: (tmp_path / "renv.lock").write_text("{}", encoding="utf-8"),
            ),
        )

        seen: list[tuple[str, str, bool]] = []

        async def on_update(stream, text, truncated):
            seen.append((stream, text, truncated))

        result = await renv_init(tmp_path, on_update=on_update)

        assert result.success is True
        # Both stdout and stderr chunks should have been forwarded.
        streams = {stream for stream, _, _ in seen}
        assert streams == {"stdout", "stderr"}

    @pytest.mark.asyncio
    async def test_failure_surfaces_stderr(self, monkeypatch, tmp_path):
        """When ``renv::init`` exits non-zero the error + stderr are
        captured for the env-panel render."""
        from strata.notebook import dependencies as deps_module

        monkeypatch.setattr(
            deps_module,
            "run_rscript_command_streaming",
            _make_fake_rscript_streaming(
                success=False,
                stderr="renv error: corrupted .Rprofile",
                error="renv::init failed (exit 1)",
            ),
        )

        result = await renv_init(tmp_path)

        assert result.success is False
        assert result.action == "r_init"
        assert "exit 1" in (result.error or "")
        assert result.operation_log is not None
        assert "corrupted .Rprofile" in result.operation_log.stderr


class TestRenvAdd:
    """``renv_add`` streams ``renv::install + renv::snapshot`` for one
    package."""

    @pytest.mark.asyncio
    async def test_rejects_invalid_package_name(self, monkeypatch, tmp_path):
        """Bad name short-circuits before any subprocess. The package
        name lands inside the snippet body, so the validator is
        load-bearing for safety, not just polish."""
        from strata.notebook import dependencies as deps_module

        async def must_not_run(*a, **kw):
            pytest.fail("must not spawn for invalid package name")

        monkeypatch.setattr(deps_module, "run_rscript_command_streaming", must_not_run)

        result = await renv_add(tmp_path, "ggplot2; rm -rf /")

        assert result.success is False
        assert result.action == "r_add"
        assert "Invalid R package name" in (result.error or "")

    @pytest.mark.asyncio
    async def test_success_calls_install_and_snapshot(self, monkeypatch, tmp_path):
        """The R snippet must call both ``renv::install`` AND
        ``renv::snapshot`` — install puts the package in the library;
        snapshot writes it to the lockfile. Dropping snapshot would
        let the on-disk library and lockfile drift."""
        from strata.notebook import dependencies as deps_module

        (tmp_path / "renv.lock").write_text('{"Packages": {}}', encoding="utf-8")
        captured: list[str] = []

        def write_new_lockfile():
            (tmp_path / "renv.lock").write_text(
                '{"Packages": {"arrow": {"Version": "14.0.0"}}}', encoding="utf-8"
            )

        monkeypatch.setattr(
            deps_module,
            "run_rscript_command_streaming",
            _make_fake_rscript_streaming(
                success=True,
                stdout="installed arrow\n",
                side_effect=write_new_lockfile,
                captured_snippets=captured,
            ),
        )

        result = await renv_add(tmp_path, "arrow")

        assert result.success is True
        assert result.action == "r_add"
        assert result.package == "arrow"
        assert result.lockfile_changed is True
        snippet = captured[0]
        assert 'renv::install("arrow")' in snippet
        assert "renv::snapshot" in snippet
        assert 'type = "all"' in snippet
        assert "prompt = FALSE" in snippet

    @pytest.mark.asyncio
    async def test_failure_surfaces_error(self, monkeypatch, tmp_path):
        """CRAN package doesn't exist / network failure: error +
        stderr land in the operation_log, lockfile_changed is False."""
        from strata.notebook import dependencies as deps_module

        (tmp_path / "renv.lock").write_text('{"Packages": {}}', encoding="utf-8")
        monkeypatch.setattr(
            deps_module,
            "run_rscript_command_streaming",
            _make_fake_rscript_streaming(
                success=False,
                stderr="package 'nonexistent' is not available",
                error="renv::install failed (exit 1)",
            ),
        )

        result = await renv_add(tmp_path, "nonexistent")

        assert result.success is False
        assert result.action == "r_add"
        assert "exit 1" in (result.error or "")
        assert result.operation_log is not None
        assert "not available" in result.operation_log.stderr
