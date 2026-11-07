"""Provenance hashing for notebook cells.

A cell's hash covers its sorted input artifact hashes, its definition key and its
environment hash, so identical computations hash identically. The formulas are
cellgraph's; ``strata.notebook.keys`` assembles a cell's inputs.
"""

from __future__ import annotations

import hashlib


def compute_source_hash(source: str, language: str = "python") -> str:
    """The definition key of a *language* cell with *source*, from cellgraph.

    Whitespace and comments do not change it; anything that changes the AST does,
    and so do the annotations that change what the cell computes (``@loop``,
    ``@write_table``, ``@output_dir``, ``@per_variant``).
    """
    from strata.notebook.graph import definition_key

    return definition_key(source, language)


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


def cell_provenance_inputs(
    input_hashes: list[str],
    mount_fingerprints: list[str],
    table_fingerprints: list[str],
    fetch_fingerprints: list[str],
    dataset_fingerprints: list[str],
) -> list[str]:
    """Every input of a cell's provenance hash besides its source and env.

    The executor keys a run on these and ``compute_staleness`` rechecks it on them: one
    missing from either leaves the cell stale forever.
    """
    return (
        input_hashes
        + mount_fingerprints
        + table_fingerprints
        + fetch_fingerprints
        + dataset_fingerprints
    )


def compute_provenance_hash(
    input_hashes: list[str],
    source_hash: str,
    env_hash: str,
) -> str:
    """SHA-256 hex digest over sorted ``input_hashes``, ``source_hash`` and ``env_hash``."""
    hasher = hashlib.sha256()
    for component in sorted(input_hashes):
        hasher.update(component.encode("utf-8"))
        hasher.update(b"\x00")
    hasher.update(source_hash.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(env_hash.encode("utf-8"))
    return hasher.hexdigest()
