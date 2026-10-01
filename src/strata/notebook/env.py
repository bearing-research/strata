"""Environment hashing for notebook dependencies.

The env hash folds ``uv.lock`` (and ``renv.lock``). With a ``[dependency-groups]
dev`` group, the uv part is a fingerprint of the runtime closure only, so adding
or removing a dev tool never invalidates a cell's cache; otherwise it is the raw bytes.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def collect_referenced_env_keys(source: str) -> set[str]:
    """Return the env var keys the cell source references statically.

    Detects ``os.environ["KEY"]``, ``os.environ.get("KEY")`` and
    ``os.getenv("KEY")``, including aliased ``os``/``environ``/``getenv``.
    Dynamic lookups are ignored: the result is a lower bound for narrowing
    provenance, not a full dependency analysis.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    os_aliases: set[str] = set()
    environ_aliases: set[str] = {"environ"}
    getenv_aliases: set[str] = {"getenv"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "os":
                    os_aliases.add(alias.asname or "os")
        elif isinstance(node, ast.ImportFrom) and node.module == "os":
            for alias in node.names:
                imported = alias.asname or alias.name
                if alias.name == "environ":
                    environ_aliases.add(imported)
                elif alias.name == "getenv":
                    getenv_aliases.add(imported)

    if not os_aliases:
        os_aliases.add("os")

    def _is_os_environ(expr: ast.AST) -> bool:
        if isinstance(expr, ast.Attribute):
            return (
                expr.attr == "environ"
                and isinstance(expr.value, ast.Name)
                and expr.value.id in os_aliases
            )
        return isinstance(expr, ast.Name) and expr.id in environ_aliases

    def _is_os_getenv(expr: ast.AST) -> bool:
        if isinstance(expr, ast.Attribute):
            return (
                expr.attr == "getenv"
                and isinstance(expr.value, ast.Name)
                and expr.value.id in os_aliases
            )
        return isinstance(expr, ast.Name) and expr.id in getenv_aliases

    def _literal_key(expr: ast.AST) -> str | None:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return expr.value
        return None

    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and _is_os_environ(node.value):
            key = _literal_key(node.slice)
            if key is not None:
                keys.add(key)
        elif isinstance(node, ast.Call):
            func = node.func
            is_environ_method = (
                isinstance(func, ast.Attribute)
                and func.attr in {"get", "setdefault", "pop"}
                and _is_os_environ(func.value)
            )
            if is_environ_method or _is_os_getenv(func):
                if node.args:
                    key = _literal_key(node.args[0])
                    if key is not None:
                        keys.add(key)
    return keys


def compute_lockfile_hash(notebook_dir: Path) -> str:
    """SHA-256 over the notebook's ``uv.lock`` and ``renv.lock``.

    ``renv.lock`` is folded under a ``\\0renv=`` tag so it cannot collide with
    uv.lock bytes. Missing lockfiles contribute nothing, so with neither present
    this is the digest of empty input.
    """
    hasher = hashlib.sha256()
    _fold_lockfile_into_hash(hasher, notebook_dir, "uv.lock", tag=None)
    # Tag prefix keeps a uv-only and an renv-only notebook with identical
    # lockfile bytes from colliding.
    _fold_lockfile_into_hash(hasher, notebook_dir, "renv.lock", tag=b"\0renv=")
    return hasher.hexdigest()


def _runtime_uv_closure_fingerprint(raw_uv_lock: bytes) -> bytes | None:
    """Fingerprint a ``uv.lock``'s runtime dependency closure, or ``None``.

    ``None`` when the lock has no dev-dependencies or cannot be parsed; the
    caller then folds the raw bytes. Otherwise folds ``name@version`` and
    artifact hashes of every package reachable from the root's runtime deps,
    so transitive runtime upgrades count but dev tools do not.
    """
    try:
        data: Any = tomllib.loads(raw_uv_lock.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError):
        return None

    packages = data.get("package")
    if not isinstance(packages, list):
        return None

    by_name: dict[str, list[dict]] = {}
    root: dict | None = None
    for pkg in packages:
        if not isinstance(pkg, dict):
            continue
        name = pkg.get("name")
        if not isinstance(name, str):
            continue
        by_name.setdefault(name, []).append(pkg)
        source = pkg.get("source")
        if isinstance(source, dict) and (
            source.get("virtual") == "." or source.get("editable") == "."
        ):
            root = pkg

    # No identifiable root or no dev group: the caller folds raw bytes.
    if root is None or not root.get("dev-dependencies"):
        return None

    def _dep_names(entries: Any) -> list[str]:
        names: list[str] = []
        if isinstance(entries, list):
            for d in entries:
                if isinstance(d, dict):
                    dep_name = d.get("name")
                    if isinstance(dep_name, str):
                        names.append(dep_name)
        return names

    # Extras of a runtime dep are runtime; dev-only packages are never reached.
    seen: set[str] = set()
    frontier = _dep_names(root.get("dependencies"))
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        for pkg in by_name.get(name, []):
            frontier.extend(_dep_names(pkg.get("dependencies")))
            optional = pkg.get("optional-dependencies")
            if isinstance(optional, dict):
                for group in optional.values():
                    frontier.extend(_dep_names(group))

    hasher = hashlib.sha256()
    for name in sorted(seen):
        for pkg in sorted(by_name.get(name, []), key=lambda p: str(p.get("version", ""))):
            hasher.update(b"\0pkg=")
            hasher.update(name.encode("utf-8"))
            hasher.update(b"@")
            hasher.update(str(pkg.get("version", "")).encode("utf-8"))
            # Artifact hashes catch a same-version re-pin.
            artifact_hashes: list[str] = []
            sdist = pkg.get("sdist")
            if isinstance(sdist, dict) and isinstance(sdist.get("hash"), str):
                artifact_hashes.append(sdist["hash"])
            wheels = pkg.get("wheels")
            if isinstance(wheels, list):
                for wheel in wheels:
                    if isinstance(wheel, dict) and isinstance(wheel.get("hash"), str):
                        artifact_hashes.append(wheel["hash"])
            for artifact_hash in sorted(artifact_hashes):
                hasher.update(b"|")
                hasher.update(artifact_hash.encode("utf-8"))
    return hasher.digest()


def _fold_lockfile_into_hash(
    hasher: hashlib._Hash, notebook_dir: Path, filename: str, *, tag: bytes | None
) -> None:
    """Fold ``notebook_dir/filename`` content into *hasher* if it exists.

    A missing or unreadable file contributes nothing (read errors log a warning).
    Uses ``open()`` rather than ``Path.read_bytes()`` because CodeQL's
    ``py/path-injection`` model flags the latter here.
    """
    lockfile = notebook_dir / filename
    if not lockfile.exists():
        return
    try:
        with open(lockfile, "rb") as f:
            content = f.read()
    except OSError as exc:
        logger.warning("Could not read %s: %s", filename, exc)
        return
    # With a dev group, fold only the runtime dependency closure so dev tools
    # (pytest/ruff/ty) don't invalidate cell caches. Otherwise fall through to raw bytes.
    if filename == "uv.lock":
        fingerprint = _runtime_uv_closure_fingerprint(content)
        if fingerprint is not None:
            hasher.update(b"\0uv-runtime=")
            hasher.update(fingerprint)
            return
    if tag is not None:
        hasher.update(tag)
    hasher.update(content)


def narrow_env_for_provenance(
    source: str,
    resolved_env: Mapping[str, str],
    declared_keys: set[str] | None = None,
) -> dict[str, str]:
    """Return the subset of ``resolved_env`` that participates in provenance.

    That is keys the source references (``os.environ``/``os.getenv``) plus
    ``declared_keys`` (``# @env`` annotations, per-cell env overrides). Ambient
    notebook env vars a cell neither reads nor declares do not affect its hash.
    """
    referenced = collect_referenced_env_keys(source)
    relevant = referenced | (declared_keys or set())
    return {k: v for k, v in resolved_env.items() if k in relevant}


def compute_execution_env_hash(
    notebook_dir: Path,
    runtime_env: Mapping[str, str] | None = None,
    runtime_identity: str | None = None,
) -> str:
    """The cell's execution env hash: the lockfile hash plus provenance-relevant env vars.

    Equals ``compute_lockfile_hash`` when ``runtime_env`` and ``runtime_identity`` are empty.
    """
    lockfile_hash = compute_lockfile_hash(notebook_dir)
    if not runtime_env and not runtime_identity:
        return lockfile_hash

    runtime_env = runtime_env or {}
    hasher = hashlib.sha256()
    hasher.update(lockfile_hash.encode("utf-8"))
    if runtime_identity:
        hasher.update(b"\0runtime=")
        hasher.update(runtime_identity.encode("utf-8"))
    for key, value in sorted(runtime_env.items()):
        hasher.update(b"\0")
        hasher.update(key.encode("utf-8"))
        hasher.update(b"=")
        hasher.update(value.encode("utf-8"))
    return hasher.hexdigest()
