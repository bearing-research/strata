"""Tests for provenance hashing."""

import hashlib

from strata.notebook.provenance import (
    compute_provenance_hash,
    compute_source_hash,
    derive_subkey,
)


def test_source_hash_stability():
    """Same source should produce same hash."""
    source = "x = 1 + 1"
    hash1 = compute_source_hash(source)
    hash2 = compute_source_hash(source)
    assert hash1 == hash2


def test_source_hash_changes_with_source():
    """Different source should produce different hash."""
    source1 = "x = 1 + 1"
    source2 = "x = 1 + 2"
    hash1 = compute_source_hash(source1)
    hash2 = compute_source_hash(source2)
    assert hash1 != hash2


def test_source_hash_ignores_cosmetic_whitespace():
    """Cosmetic whitespace / blank line / comment edits must NOT invalidate.

    The hash is taken over the AST's canonical unparse form, so reformatting
    a cell (autoformatter, trailing newlines, extra spacing around
    operators) keeps the cached artifact. Only semantic changes invalidate.
    """
    variants = [
        "x = 1 + 1",
        "x = 1 +  1",  # double space
        "x = 1 + 1\n",  # trailing newline
        "x = 1 + 1\n\n\n",  # trailing blank lines
        "x = 1 + 1   ",  # trailing spaces
        "# intro comment\nx = 1 + 1",  # added comment
    ]
    hashes = {compute_source_hash(v) for v in variants}
    assert len(hashes) == 1


def test_source_hash_detects_semantic_change():
    """Swapping a literal value changes the hash."""
    assert compute_source_hash("x = 1 + 1") != compute_source_hash("x = 1 + 2")


def test_provenance_hash_stability():
    """Same inputs should produce same provenance hash."""
    input_hashes = ["hash1", "hash2"]
    source_hash = compute_source_hash("x = 1")
    env_hash = compute_source_hash("env")

    hash1 = compute_provenance_hash(input_hashes, source_hash, env_hash)
    hash2 = compute_provenance_hash(input_hashes, source_hash, env_hash)

    assert hash1 == hash2


def test_provenance_hash_ordering_invariance():
    """Input order should not affect provenance hash."""
    input_hashes1 = ["hash1", "hash2", "hash3"]
    input_hashes2 = ["hash3", "hash1", "hash2"]
    source_hash = compute_source_hash("x = 1")
    env_hash = compute_source_hash("env")

    hash1 = compute_provenance_hash(input_hashes1, source_hash, env_hash)
    hash2 = compute_provenance_hash(input_hashes2, source_hash, env_hash)

    assert hash1 == hash2


def test_provenance_hash_changes_with_source():
    """Source hash change should affect provenance hash."""
    input_hashes = ["hash1"]
    source1 = compute_source_hash("x = 1")
    source2 = compute_source_hash("x = 2")
    env_hash = compute_source_hash("env")

    hash1 = compute_provenance_hash(input_hashes, source1, env_hash)
    hash2 = compute_provenance_hash(input_hashes, source2, env_hash)

    assert hash1 != hash2


def test_provenance_hash_changes_with_env():
    """Env hash change should affect provenance hash."""
    input_hashes = ["hash1"]
    source_hash = compute_source_hash("x = 1")
    env1 = compute_source_hash("env1")
    env2 = compute_source_hash("env2")

    hash1 = compute_provenance_hash(input_hashes, source_hash, env1)
    hash2 = compute_provenance_hash(input_hashes, source_hash, env2)

    assert hash1 != hash2


def test_provenance_hash_changes_with_inputs():
    """Input change should affect provenance hash."""
    source_hash = compute_source_hash("x = 1")
    env_hash = compute_source_hash("env")

    hash1 = compute_provenance_hash(["hash1"], source_hash, env_hash)
    hash2 = compute_provenance_hash(["hash2"], source_hash, env_hash)

    assert hash1 != hash2


def test_provenance_hash_empty_inputs():
    """Empty inputs should be valid."""
    source_hash = compute_source_hash("x = 1")
    env_hash = compute_source_hash("env")

    hash1 = compute_provenance_hash([], source_hash, env_hash)
    hash2 = compute_provenance_hash([], source_hash, env_hash)

    assert hash1 == hash2


# ---------------------------------------------------------------------------
# derive_subkey


def test_derive_subkey_matches_legacy_inline_form():
    """Wire stability: derive_subkey must produce byte-identical output to
    the inline ``hashlib.sha256(f"{parent}:{label}".encode()).hexdigest()``
    pattern callers used before the extraction. Changing the byte format
    would invalidate every cached artifact keyed off a derived hash."""
    parent = "a" * 64
    expected = hashlib.sha256(f"{parent}:varname".encode()).hexdigest()
    assert derive_subkey(parent, "varname") == expected


def test_derive_subkey_multi_label_matches_legacy_inline_form():
    """Multi-label variant: same byte-identity invariant for the loop
    iter-provenance shape ``f"{parent1}:{parent2}:iter={k}"``."""
    expected = hashlib.sha256(b"p1:p2:iter=3").hexdigest()
    assert derive_subkey("p1", "p2", "iter=3") == expected


def test_derive_subkey_stable():
    """Repeat calls with identical args return identical hashes."""
    assert derive_subkey("parent", "x") == derive_subkey("parent", "x")


def test_derive_subkey_label_distinguishes_outputs():
    """Different labels produce different hashes (the whole point — two
    output variables of the same cell get distinct artifact keys)."""
    parent = "common"
    assert derive_subkey(parent, "x") != derive_subkey(parent, "y")


def test_derive_subkey_parent_distinguishes_cells():
    """Different parents produce different hashes (two cells with the
    same output name still get distinct artifact keys)."""
    assert derive_subkey("cell_a", "result") != derive_subkey("cell_b", "result")


def test_derive_subkey_zero_labels_is_just_parent_hash():
    """No labels degenerates to hashing the parent alone — useful as the
    identity element when a caller iterates over an optional label list."""
    parent = "abc"
    assert derive_subkey(parent) == hashlib.sha256(parent.encode()).hexdigest()
