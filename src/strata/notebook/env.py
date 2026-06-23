"""Environment hashing for notebook dependencies.

The cell-provenance env hash folds ``uv.lock`` (+ ``renv.lock``). A notebook with
no dev-dependencies hashes the **raw ``uv.lock`` bytes** (the historical
behavior — byte-identical, so existing caches are untouched). When a
``[dependency-groups] dev`` group is present, the uv contribution becomes a
fingerprint of just the **runtime dependency closure** instead, so adding or
removing a dev tool (pytest / ruff / ty / mypy) — which doesn't change what a
cell *computes* — never invalidates the cell's cache.
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
    """Return the set of env var keys the cell source references statically.

    Detects the common access patterns:
        os.environ["KEY"]               Subscript on ``os.environ``
        os.environ.get("KEY", ...)      Method call on ``os.environ``
        os.getenv("KEY", ...)           Top-level ``os.getenv``

    Also recognises the same patterns when ``environ``, ``getenv``, or
    ``os`` have been aliased (``from os import environ, getenv`` / ``import
    os as o``). Dynamic lookups (``os.environ[key]`` with a variable) are
    ignored — the returned set is a lower bound used for narrowing
    provenance, not an exhaustive dependency analysis.
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
    """Compute SHA-256 hash over the notebook's lockfiles.

    Folds together:

    * ``uv.lock`` — Python dependency pins (the original behaviour).
    * ``renv.lock`` — R dependency pins, when present. Contributes
      under a ``\\0renv=`` tag so a renv.lock with bytes identical to
      some hypothetical uv.lock can't collide.

    Backward compatibility: when ``renv.lock`` is absent the result
    is byte-identical to the pre-#59 hash (single ``sha256(uv.lock)``
    call), so Python-only notebooks see no cache invalidation from
    this change.

    Why include renv.lock here rather than a new R-specific helper:
    ``compute_execution_env_hash`` calls this once per provenance
    pass and doesn't otherwise know the cell's language. Threading
    language-awareness through would expand the call sites for no
    benefit — the *only* additional input is renv.lock content,
    which is cheap to read whether or not the notebook has R cells.

    If renv.lock doesn't exist (e.g., notebook has no R deps) and
    uv.lock doesn't either, return the sentinel empty-string hash.

    Args:
        notebook_dir: Path to notebook directory

    Returns:
        SHA-256 hex digest folded over present lockfiles.
    """
    hasher = hashlib.sha256()
    _fold_lockfile_into_hash(hasher, notebook_dir, "uv.lock", tag=None)
    # Tag prefix prevents an exotic collision between a uv-only
    # notebook and a renv-only one whose lockfiles happen to share
    # bytes. The tag itself is fixed bytes, not derived from
    # notebook state.
    _fold_lockfile_into_hash(hasher, notebook_dir, "renv.lock", tag=b"\0renv=")
    return hasher.hexdigest()


def _runtime_uv_closure_fingerprint(raw_uv_lock: bytes) -> bytes | None:
    """Fingerprint a ``uv.lock``'s runtime dependency closure, or ``None``.

    Returns ``None`` when the lock declares **no** dev-dependencies (the caller
    then folds the raw bytes — byte-identical to the historical hash, so notebooks
    without dev tooling are completely unaffected) or cannot be parsed (safe
    fallback to raw bytes).

    Otherwise it walks the resolution graph from the root project's **runtime**
    direct deps (its ``dependencies`` — *not* ``[package.dev-dependencies]``) and
    folds each reachable package's ``name@version`` plus its artifact hashes. So
    the digest tracks runtime content — including transitive runtime upgrades a
    dev install might force — while being invariant to the dev tools themselves.
    """
    try:
        # tomllib yields dynamically-typed nested structures; treat as Any and
        # guard each access with isinstance below.
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

    # No identifiable root, or no dev group to exclude → let the caller fold raw
    # bytes (the historical, byte-identical behavior).
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

    # Seed with the root's RUNTIME direct deps, then walk each reached package's
    # own dependencies + optional-dependencies (extras of a runtime dep are
    # runtime). Dev-only packages are never reached → excluded.
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
            # Artifact hashes pin content — catches a same-version re-pin.
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

    Quietly skips on missing file or read error — callers (the
    aggregate ``compute_lockfile_hash``) treat a missing lockfile as
    "no contribution", and the warning is informational only.

    Uses ``open() + read()`` rather than ``Path.read_bytes()``
    because CodeQL's ``py/path-injection`` taint model flags the
    latter on a Path constructed from a function argument even when
    the argument is internal trusted state (here:
    ``session.path``). The ``open()`` form matches the pre-existing
    helper signature CodeQL is already happy with.
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
    # uv.lock: when a dev group is present, fold a fingerprint of only the
    # *runtime* dependency closure instead of the raw bytes, so dev tools
    # (pytest/ruff/ty) don't invalidate cell caches. No dev group (or a parse
    # failure) → fall through to the raw-bytes fold, byte-identical to the
    # historical hash (no re-hash for existing/non-dev notebooks).
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

    Relevant keys are the union of:

    * keys referenced in the cell source via ``os.environ[...]`` /
      ``os.environ.get(...)`` / ``os.getenv(...)``, and
    * keys in ``declared_keys`` — explicit cell-level declarations such
      as ``# @env KEY=value`` annotations or persisted per-cell env
      overrides in ``notebook.toml``.

    Notebook-level ambient env vars that a cell neither references nor
    declares do not influence its hash, so adding an API key at the
    notebook level does not invalidate cells that do not use it.
    """
    referenced = collect_referenced_env_keys(source)
    relevant = referenced | (declared_keys or set())
    return {k: v for k, v in resolved_env.items() if k in relevant}


def compute_execution_env_hash(
    notebook_dir: Path,
    runtime_env: Mapping[str, str] | None = None,
    runtime_identity: str | None = None,
) -> str:
    """Compute the effective execution environment hash for a cell.

    This combines the notebook lockfile hash with any persisted or annotated
    runtime environment variables that should participate in provenance.

    If ``runtime_env`` and ``runtime_identity`` are empty, this is identical
    to ``compute_lockfile_hash``.
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
