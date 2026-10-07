"""Helpers for notebook Python-version selection and persistence."""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, metadata
from pathlib import Path

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version


def current_python_minor() -> str:
    """Return the current interpreter's ``major.minor`` version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def normalize_python_minor(version: str) -> str:
    """Validate and canonicalize a Python ``major.minor`` string.

    Rejects patch (``3.12.0``), pre/post/dev (``3.12rc1``) and a leading ``v``:
    ``major.minor`` is what ``uv --python`` accepts as a minor-line request, and a
    patch version would pin or fail downstream.
    """
    stripped = version.strip()
    try:
        parsed = Version(stripped)
    except InvalidVersion as exc:
        raise ValueError(
            "Python version must use major.minor format like '3.12' or '3.13'"
        ) from exc
    canonical = f"{parsed.major}.{parsed.minor}"
    if canonical != stripped:
        raise ValueError("Python version must use major.minor format like '3.12' or '3.13'")
    return canonical


def format_requires_python(version: str) -> str:
    """Return a ``requires-python`` spec (``==X.Y.*``) that pins one minor line."""
    return f"=={normalize_python_minor(version)}.*"


def infer_requested_python_minor(requires_python: str | None) -> str | None:
    """Extract ``major.minor`` from a notebook ``requires-python``.

    Only the canonical ``==X.Y.*`` is accepted, since notebooks pin exactly one
    minor. Returns ``None`` for an empty or malformed spec, or any other form.
    """
    if not requires_python:
        return None
    try:
        spec = SpecifierSet(requires_python.strip())
    except InvalidSpecifier:
        return None
    clauses = list(spec)
    if len(clauses) != 1:
        return None
    clause = clauses[0]
    if clause.operator != "==" or not clause.version.endswith(".*"):
        return None
    try:
        parsed = Version(clause.version.removesuffix(".*"))
    except InvalidVersion:
        return None
    if len(parsed.release) != 2:
        return None
    return f"{parsed.major}.{parsed.minor}"


def read_requested_python_minor(notebook_dir: Path) -> str | None:
    """Read the requested Python minor version from a notebook pyproject."""
    pyproject_path = Path(notebook_dir) / "pyproject.toml"
    if not pyproject_path.exists():
        return None
    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        return None
    project = data.get("project")
    if not isinstance(project, dict):
        return None
    requires_python = project.get("requires-python")
    if not isinstance(requires_python, str):
        return None
    return infer_requested_python_minor(requires_python)


def discover_installed_python_minors() -> list[str]:
    """Return installed ``major.minor`` versions uv reports, within Strata's ``requires-python``.

    Falls back to ``[current_python_minor()]`` on any failure. Probed once per process.
    """
    return list(_probe_installed_python_minors())


# StrataConfig.load runs this as a default on every load (each notebook open and
# staleness pass outside the server), and the uv subprocess costs seconds under load.
@functools.cache
def _probe_installed_python_minors() -> tuple[str, ...]:
    current = current_python_minor()
    fallback = (current,)

    uv = shutil.which("uv")
    if uv is None:
        return fallback
    try:
        completed = subprocess.run(  # noqa: S603 (uv resolved via shutil.which)
            [uv, "python", "list", "--only-installed", "--output-format", "json"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        entries = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return fallback
    if not isinstance(entries, list):
        return fallback

    try:
        requires = metadata("strata-notebook").get("Requires-Python") or ">=3.12"
    except PackageNotFoundError:
        requires = ">=3.12"
    spec = SpecifierSet(requires)

    minors: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw = entry.get("version")
        if not isinstance(raw, str):
            continue
        try:
            parsed = Version(raw)
        except InvalidVersion:
            continue
        minor = f"{parsed.major}.{parsed.minor}"
        if minor in seen or not spec.contains(minor):
            continue
        seen.add(minor)
        minors.append(minor)

    if current not in seen and spec.contains(current):
        minors.insert(0, current)
    return tuple(minors) or fallback


def read_venv_runtime_python_version(python_executable: Path) -> str | None:
    """Read ``major.minor.micro`` from a venv ``pyvenv.cfg`` when available."""
    python_path = Path(python_executable)
    config_path = python_path.parent.parent / "pyvenv.cfg"
    if not config_path.exists():
        return None
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for raw_line in text.splitlines():
        key, sep, value = raw_line.partition("=")
        if sep != "=" or key.strip() != "version_info":
            continue
        version = value.strip()
        try:
            parsed = Version(version)
        except InvalidVersion:
            return None
        return version if len(parsed.release) == 3 else None
    return None
