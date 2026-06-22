"""Tests for notebook Python-version parsing / persistence helpers."""

from __future__ import annotations

import pytest

from strata.notebook import python_versions as pv


class TestNormalizePythonMinor:
    def test_valid_major_minor(self):
        assert pv.normalize_python_minor("3.12") == "3.12"
        assert pv.normalize_python_minor(" 3.13 ") == "3.13"

    @pytest.mark.parametrize("bad", ["3.12.0", "3.12rc1", "v3.12", "3"])
    def test_non_major_minor_rejected(self, bad):
        with pytest.raises(ValueError, match="major.minor"):
            pv.normalize_python_minor(bad)

    def test_garbage_rejected(self):
        with pytest.raises(ValueError, match="major.minor"):
            pv.normalize_python_minor("not-a-version")


class TestFormatRequiresPython:
    def test_emits_wildcard_pin(self):
        assert pv.format_requires_python("3.12") == "==3.12.*"

    def test_rejects_patch(self):
        with pytest.raises(ValueError):
            pv.format_requires_python("3.12.1")


class TestInferRequestedPythonMinor:
    def test_canonical_wildcard(self):
        assert pv.infer_requested_python_minor("==3.12.*") == "3.12"

    @pytest.mark.parametrize("value", [None, ""])
    def test_empty_returns_none(self, value):
        assert pv.infer_requested_python_minor(value) is None

    def test_invalid_specifier_returns_none(self):
        assert pv.infer_requested_python_minor("not-a-spec") is None

    def test_multiple_clauses_returns_none(self):
        assert pv.infer_requested_python_minor(">=3.12,<3.13") is None

    @pytest.mark.parametrize("value", [">=3.12", "~=3.12", "==3.12"])
    def test_non_wildcard_equals_returns_none(self, value):
        assert pv.infer_requested_python_minor(value) is None

    @pytest.mark.parametrize("value", ["==3.*", "==3.12.5.*"])
    def test_wrong_release_length_returns_none(self, value):
        assert pv.infer_requested_python_minor(value) is None

    def test_invalid_version_in_wildcard_returns_none(self):
        assert pv.infer_requested_python_minor("==abc.*") is None


class TestReadRequestedPythonMinor:
    def test_missing_pyproject(self, tmp_path):
        assert pv.read_requested_python_minor(tmp_path) is None

    def test_malformed_toml(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("this is { not valid toml")
        assert pv.read_requested_python_minor(tmp_path) is None

    def test_no_project_table(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[tool.strata]\nx = 1\n")
        assert pv.read_requested_python_minor(tmp_path) is None

    def test_requires_python_not_string(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'nb'\n")
        assert pv.read_requested_python_minor(tmp_path) is None

    def test_valid_requires_python(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname = 'nb'\nrequires-python = '==3.12.*'\n"
        )
        assert pv.read_requested_python_minor(tmp_path) == "3.12"


class TestReadVenvRuntimePythonVersion:
    def _venv(self, tmp_path, cfg_contents=None):
        (tmp_path / "bin").mkdir()
        python = tmp_path / "bin" / "python"
        python.write_text("")
        if cfg_contents is not None:
            (tmp_path / "pyvenv.cfg").write_text(cfg_contents)
        return python

    def test_missing_config(self, tmp_path):
        python = self._venv(tmp_path)
        assert pv.read_venv_runtime_python_version(python) is None

    def test_full_version(self, tmp_path):
        python = self._venv(tmp_path, "home = /usr\nversion_info = 3.12.5\n")
        assert pv.read_venv_runtime_python_version(python) == "3.12.5"

    def test_no_version_info_line(self, tmp_path):
        python = self._venv(tmp_path, "home = /usr\nfoo = bar\n")
        assert pv.read_venv_runtime_python_version(python) is None

    def test_two_component_version_rejected(self, tmp_path):
        python = self._venv(tmp_path, "version_info = 3.12\n")
        assert pv.read_venv_runtime_python_version(python) is None

    def test_garbage_version_rejected(self, tmp_path):
        python = self._venv(tmp_path, "version_info = not-a-version\n")
        assert pv.read_venv_runtime_python_version(python) is None


class TestDiscoverInstalledPythonMinors:
    def test_falls_back_when_uv_missing(self, monkeypatch):
        monkeypatch.setattr(pv.shutil, "which", lambda _: None)
        assert pv.discover_installed_python_minors() == [pv.current_python_minor()]

    def test_falls_back_on_subprocess_error(self, monkeypatch):
        monkeypatch.setattr(pv.shutil, "which", lambda _: "/usr/bin/uv")

        def boom(*a, **k):
            raise OSError("uv exploded")

        monkeypatch.setattr(pv.subprocess, "run", boom)
        assert pv.discover_installed_python_minors() == [pv.current_python_minor()]

    def test_falls_back_when_output_not_a_list(self, monkeypatch):
        monkeypatch.setattr(pv.shutil, "which", lambda _: "/usr/bin/uv")
        monkeypatch.setattr(
            pv.subprocess,
            "run",
            lambda *a, **k: _Completed('{"not": "a list"}'),
        )
        assert pv.discover_installed_python_minors() == [pv.current_python_minor()]

    def test_parses_dedupes_and_prepends_current(self, monkeypatch):
        monkeypatch.setattr(pv.shutil, "which", lambda _: "/usr/bin/uv")
        # Two 3.13 patches (dedupe), one 3.99 (kept by >=3.12 spec), a bad
        # version, a non-dict entry, and a dict without a version string.
        payload = (
            '[{"version": "3.13.1"}, {"version": "3.13.2"}, '
            '{"version": "3.99.0"}, {"version": "bad"}, "not-a-dict", {"no": "version"}]'
        )
        monkeypatch.setattr(pv.subprocess, "run", lambda *a, **k: _Completed(payload))

        result = pv.discover_installed_python_minors()
        assert result[0] == pv.current_python_minor()  # current prepended
        assert "3.13" in result
        assert "3.99" in result
        assert result.count("3.13") == 1  # deduped

    def test_current_version_not_duplicated_when_already_listed(self, monkeypatch):
        monkeypatch.setattr(pv.shutil, "which", lambda _: "/usr/bin/uv")
        current = pv.current_python_minor()
        monkeypatch.setattr(
            pv.subprocess,
            "run",
            lambda *a, **k: _Completed(f'[{{"version": "{current}.0"}}]'),
        )
        result = pv.discover_installed_python_minors()
        assert result.count(current) == 1  # not prepended again

    def test_uses_default_spec_when_metadata_missing(self, monkeypatch):
        from importlib.metadata import PackageNotFoundError

        monkeypatch.setattr(pv.shutil, "which", lambda _: "/usr/bin/uv")
        monkeypatch.setattr(
            pv.subprocess, "run", lambda *a, **k: _Completed('[{"version": "3.13.0"}]')
        )

        def _missing(_name):
            raise PackageNotFoundError("strata-notebook")

        monkeypatch.setattr(pv, "metadata", _missing)
        result = pv.discover_installed_python_minors()
        assert "3.13" in result  # >=3.12 default spec still admits 3.13


class _Completed:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, stdout: str):
        self.stdout = stdout
        self.returncode = 0
