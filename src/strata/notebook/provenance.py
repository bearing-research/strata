"""Provenance hashing for notebook cells.

Provenance hashing enables cache deduplication by computing a deterministic
hash of:
1. The sorted input artifact hashes (from upstream cells)
2. The cell source code (normalized)
3. The runtime environment hash (lockfile)

This ensures identical computations always produce the same hash and can
be cached.
"""

from __future__ import annotations

import ast
import hashlib


def _normalize_source_for_hash(source: str) -> str:
    """Return a canonical form of *source* for provenance hashing.

    Normalization must reject cosmetic edits that don't change
    behavior (blank lines, trailing whitespace, comments, single vs
    double quotes, ``1+2`` vs ``1 + 2``) while preserving any change
    that could affect execution. We get that for free by round-tripping
    through the AST: ``ast.parse`` tolerates all whitespace as long as
    it's syntactically valid, and ``ast.unparse`` emits a stable
    canonical form keyed only to the semantic tree.

    If the source can't be parsed (user hit Run on an incomplete
    edit), fall back to a weaker normalization that still absorbs the
    most common edits: trailing whitespace per line and leading /
    trailing blank lines.
    """
    try:
        tree = ast.parse(source)
        return ast.unparse(tree)
    except SyntaxError:
        lines = [line.rstrip() for line in source.splitlines()]
        return "\n".join(lines).strip()


def compute_source_hash(source: str) -> str:
    """Compute SHA-256 hash of a semantically-normalized cell source.

    Whitespace, blank lines, and comments do NOT invalidate the cache.
    Anything that changes the parsed AST does — variable renames,
    literal value changes, control-flow edits, etc.

    Args:
        source: Cell source code

    Returns:
        SHA-256 hex digest
    """
    normalized = _normalize_source_for_hash(source)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def derive_subkey(parent_hash: str, *labels: str) -> str:
    """Derive a sub-hash from a parent provenance hash plus one or more labels.

    Returns ``sha256("parent_hash:label1:label2:...")``. Used to namespace
    per-variable, per-display, and per-iteration provenance off a cell's
    main provenance hash, so two cells that share a provenance hash still
    produce distinct artifact keys for their distinct outputs.

    The byte format (colon-joined, no separator after the last component)
    is wire-stable — changing it would invalidate every existing cached
    artifact keyed off a derived hash.

    Args:
        parent_hash: The originating provenance hash (or any seed value).
        *labels: One or more discriminating labels appended after the parent.

    Returns:
        SHA-256 hex digest of ``parent_hash:label1:label2:...``.
    """
    pieces = [parent_hash, *labels]
    return hashlib.sha256(":".join(pieces).encode()).hexdigest()


def compute_provenance_hash(
    input_hashes: list[str],
    source_hash: str,
    env_hash: str,
) -> str:
    """Compute the provenance hash for a cell execution.

    The hash uniquely identifies a computation based on:
    1. Content hashes of all input artifacts (sorted for determinism)
    2. The cell source code hash
    3. The runtime environment hash (lockfile)

    Args:
        input_hashes: Hashes of upstream artifacts this cell consumes.
                     Will be sorted for deterministic ordering.
        source_hash: SHA-256 of cell source code
        env_hash: SHA-256 of runtime lockfile dependencies

    Returns:
        SHA-256 hex digest of the combined provenance
    """
    # Sort input hashes for deterministic ordering
    sorted_inputs = sorted(input_hashes)

    # Combine all components
    hasher = hashlib.sha256()

    # Add sorted input hashes
    for h in sorted_inputs:
        hasher.update(h.encode("utf-8"))
        hasher.update(b"\x00")  # Separator

    # Add source hash
    hasher.update(source_hash.encode("utf-8"))
    hasher.update(b"\x00")

    # Add environment hash
    hasher.update(env_hash.encode("utf-8"))

    return hasher.hexdigest()
