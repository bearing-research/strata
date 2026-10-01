"""Tests for missing-module detection, notebook export, and per-notebook dependency locks."""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from strata.notebook.dependencies import _get_notebook_lock
from strata.notebook.executor import _detect_missing_module
from tests.notebook.e2e_fixtures import (
    NotebookBuilder,
    create_test_app,
    open_notebook_session,
)


class TestDetectMissingModule:
    """_detect_missing_module extracts (language, package) from cell errors."""

    def test_simple_module(self):
        assert _detect_missing_module("ModuleNotFoundError: No module named 'requests'", "") == (
            "python",
            "requests",
        )

    def test_submodule(self):
        """Submodule import extracts top-level."""
        assert _detect_missing_module("ModuleNotFoundError: No module named 'numpy.core'", "") == (
            "python",
            "numpy",
        )

    def test_double_quotes(self):
        assert _detect_missing_module('ModuleNotFoundError: No module named "pandas"', "") == (
            "python",
            "pandas",
        )

    def test_from_stderr(self):
        assert _detect_missing_module("", "ModuleNotFoundError: No module named 'flask'") == (
            "python",
            "flask",
        )

    def test_pil_to_pillow(self):
        assert _detect_missing_module("ModuleNotFoundError: No module named 'PIL'", "") == (
            "python",
            "Pillow",
        )

    def test_sklearn_to_scikit(self):
        assert _detect_missing_module("ModuleNotFoundError: No module named 'sklearn'", "") == (
            "python",
            "scikit-learn",
        )

    def test_cv2_to_opencv(self):
        assert _detect_missing_module("ModuleNotFoundError: No module named 'cv2'", "") == (
            "python",
            "opencv-python",
        )

    def test_yaml_to_pyyaml(self):
        assert _detect_missing_module("ModuleNotFoundError: No module named 'yaml'", "") == (
            "python",
            "pyyaml",
        )

    def test_no_match(self):
        assert _detect_missing_module("TypeError: foo", "") is None

    def test_empty(self):
        assert _detect_missing_module("", "") is None

    # R cells emit a different shape, so the install button dispatches to
    # install.packages() rather than uv add.

    def test_r_library_missing(self):
        """``library(arrow)`` failure surfaces as an R suggestion."""
        msg = "Error in library(arrow) : there is no package called 'arrow'"
        assert _detect_missing_module(msg, "") == ("r", "arrow")

    def test_r_require_namespace_missing(self):
        """``requireNamespace`` produces the same error wording."""
        msg = "Error: there is no package called 'tibble'"
        assert _detect_missing_module(msg, "") == ("r", "tibble")

    def test_r_unicode_curly_quotes(self):
        """Some R locales emit ``‘pkg’`` instead of ``'pkg'``; both must be detected, since the
        locale varies across CI runners.
        """
        msg = "Error in library(ggplot2) : there is no package called ‘ggplot2’"
        assert _detect_missing_module(msg, "") == ("r", "ggplot2")

    def test_r_error_from_stderr(self):
        """Rscript writes errors to stderr, so they are detected there too."""
        stderr = "Error: there is no package called 'arrow'\nExecution halted"
        assert _detect_missing_module("", stderr) == ("r", "arrow")

    def test_python_match_wins_over_r_when_both_present(self):
        """In a mixed-language error the Python match wins, since that is the cell that failed."""
        msg = (
            "ModuleNotFoundError: No module named 'pandas'\n"
            "Error in library(arrow): there is no package called 'arrow'"
        )
        assert _detect_missing_module(msg, "") == ("python", "pandas")


class TestExportEndpoint:
    """GET /{id}/export produces a valid zip."""

    @pytest.fixture
    def setup(self):
        app = create_test_app()
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as tmpdir:
            yield client, Path(tmpdir)

    def test_export_contains_expected_files(self, setup):
        """Export zip contains notebook.toml, pyproject.toml, provenance.json."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1")

        with open_notebook_session(client, nb.path) as (sid, session):
            resp = client.get(f"/v1/notebooks/{sid}/export")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "application/zip"

            buf = io.BytesIO(resp.content)
            with zipfile.ZipFile(buf) as zf:
                names = zf.namelist()
                assert "notebook.toml" in names
                assert "pyproject.toml" in names
                assert "provenance.json" in names

    def test_export_contains_cells(self, setup):
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1").add_cell("c2", "y = 2")

        with open_notebook_session(client, nb.path) as (sid, session):
            resp = client.get(f"/v1/notebooks/{sid}/export")
            buf = io.BytesIO(resp.content)
            with zipfile.ZipFile(buf) as zf:
                names = zf.namelist()
                assert "cells/c1.py" in names
                assert "cells/c2.py" in names

    def test_export_provenance_json_valid(self, setup):
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1")

        with open_notebook_session(client, nb.path) as (sid, session):
            resp = client.get(f"/v1/notebooks/{sid}/export")
            buf = io.BytesIO(resp.content)
            with zipfile.ZipFile(buf) as zf:
                prov = json.loads(zf.read("provenance.json"))
                assert "notebook_id" in prov
                assert "lockfile_hash" in prov
                assert "dag" in prov
                assert "cells" in prov
                assert "c1" in prov["cells"]
                assert "source_hash" in prov["cells"]["c1"]

    def test_export_includes_uv_lock(self, setup):
        client, tmp = setup
        nb = NotebookBuilder(tmp)

        with open_notebook_session(client, nb.path) as (sid, session):
            resp = client.get(f"/v1/notebooks/{sid}/export")
            buf = io.BytesIO(resp.content)
            with zipfile.ZipFile(buf) as zf:
                names = zf.namelist()
                assert "uv.lock" in names

    def test_export_404_for_unknown(self, setup):
        client, tmp = setup
        resp = client.get("/v1/notebooks/nonexistent/export")
        assert resp.status_code == 404


class TestConcurrentLock:
    """Per-notebook locking for concurrent uv operations."""

    def test_same_dir_gets_same_lock(self, tmp_path):
        lock1 = _get_notebook_lock(tmp_path)
        lock2 = _get_notebook_lock(tmp_path)
        assert lock1 is lock2

    def test_different_dirs_get_different_locks(self, tmp_path):
        dir1 = tmp_path / "nb1"
        dir2 = tmp_path / "nb2"
        dir1.mkdir()
        dir2.mkdir()
        lock1 = _get_notebook_lock(dir1)
        lock2 = _get_notebook_lock(dir2)
        assert lock1 is not lock2
