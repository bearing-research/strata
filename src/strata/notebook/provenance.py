"""Provenance hashing for notebook cells.

A cell's hash covers its sorted input artifact hashes, its normalized source
and its environment hash, so identical computations hash identically.
"""

from __future__ import annotations

import ast
import hashlib


def _normalize_source_for_hash(source: str) -> str:
    """Return a canonical form of *source* for provenance hashing.

    Round-trips through the AST so cosmetic edits (whitespace, comments, quote
    style) hash the same. Unparseable source falls back to stripping trailing
    whitespace per line and leading/trailing blank lines.
    """
    try:
        tree = ast.parse(source)
        return ast.unparse(tree)
    except SyntaxError:
        lines = [line.rstrip() for line in source.splitlines()]
        return "\n".join(lines).strip()


def compute_source_hash(source: str) -> str:
    """SHA-256 hex digest of the normalized cell source.

    Whitespace and comments do not change it; anything that changes the AST does.
    """
    normalized = _normalize_source_for_hash(source)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def derive_subkey(parent_hash: str, *labels: str) -> str:
    """Derive ``sha256("parent_hash:label1:label2:...")`` from a provenance hash.

    Namespaces per-variable, per-display and per-iteration keys off a cell's
    hash. The byte format is wire-stable: changing it invalidates every cached
    artifact keyed off a derived hash.
    """
    pieces = [parent_hash, *labels]
    return hashlib.sha256(":".join(pieces).encode()).hexdigest()


def safe_filename_stem(variable_name: str) -> str:
    """Case-collision-proof filename stem for a per-variable blob.

    Names that differ only in case (``Widget`` and ``widget``) would share a file
    on a case-insensitive filesystem, so a mixed-case name gets a short hash of
    itself appended; all-lowercase names are returned unchanged. Every writer
    and reader of per-variable blob filenames must use this.
    """
    if variable_name != variable_name.lower():
        return f"{variable_name}-{hashlib.sha256(variable_name.encode()).hexdigest()[:8]}"
    return variable_name


def compute_provenance_hash(
    input_hashes: list[str],
    source_hash: str,
    env_hash: str,
) -> str:
    """SHA-256 hex digest over sorted ``input_hashes``, ``source_hash`` and ``env_hash``."""
    sorted_inputs = sorted(input_hashes)

    hasher = hashlib.sha256()

    for h in sorted_inputs:
        hasher.update(h.encode("utf-8"))
        hasher.update(b"\x00")

    hasher.update(source_hash.encode("utf-8"))
    hasher.update(b"\x00")

    hasher.update(env_hash.encode("utf-8"))

    return hasher.hexdigest()
