"""Tests for provenance hashing."""

import hashlib

from strata.notebook.provenance import (
    compute_provenance_hash,
    compute_source_hash,
    derive_subkey,
)


def test_source_hash_stability():
    source = "x = 1 + 1"
    hash1 = compute_source_hash(source)
    hash2 = compute_source_hash(source)
    assert hash1 == hash2


def test_source_hash_changes_with_source():
    source1 = "x = 1 + 1"
    source2 = "x = 1 + 2"
    hash1 = compute_source_hash(source1)
    hash2 = compute_source_hash(source2)
    assert hash1 != hash2


def test_source_hash_ignores_cosmetic_whitespace():
    """Reformatting, blank lines and comments must not invalidate the cache.

    The hash covers the AST's canonical unparse, so only semantic changes count.
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


def test_a_plain_cells_source_hash_is_pinned_to_a_known_value():
    """Annotations outside the loop fingerprint stay out of the hash, so non-loop cells keep
    their cached results.
    """
    source = "# @name totals\n# @env MODE=fast\nx = 1  # one\n"
    assert (
        compute_source_hash(source)
        == "8ff436def1451285599a1b1ad70800493b8dcafde2912e1a38345633054e4c26"
    )


def test_loop_parameters_change_the_source_hash():
    """``@loop`` and ``@loop_until`` live in comments but change what the cell computes."""
    body = "state = state + 1\n"
    variants = [
        "# @loop max_iter=3 carry=state\n",
        "# @loop max_iter=1 carry=state\n",
        "# @loop max_iter=3 carry=other\n",
        "# @loop max_iter=3 carry=state start_from=hill@iter=1\n",
        "# @loop max_iter=3 carry=state start_from=hill@iter=2\n",
        "# @loop max_iter=3 carry=state\n# @loop_until state > 2\n",
        "# @loop max_iter=3 carry=state\n# @loop_until state > 1\n",
        "",
    ]
    hashes = {compute_source_hash(header + body) for header in variants}
    assert len(hashes) == len(variants)
    # Only the parsed values count: spacing and order inside the directive do not.
    assert compute_source_hash("# @loop max_iter=3 carry=state\n" + body) == compute_source_hash(
        "#  @loop carry=state   max_iter=3\n" + body
    )


def test_provenance_hash_stability():
    input_hashes = ["hash1", "hash2"]
    source_hash = compute_source_hash("x = 1")
    env_hash = compute_source_hash("env")

    hash1 = compute_provenance_hash(input_hashes, source_hash, env_hash)
    hash2 = compute_provenance_hash(input_hashes, source_hash, env_hash)

    assert hash1 == hash2


def test_provenance_hash_is_pinned_to_a_known_value():
    """The provenance hash is every cache's key, so pin its value, not just determinism.

    Changing it silently invalidates every cached artifact in every store,
    including published ones. If this fails, make that decision deliberately.
    """
    assert (
        compute_provenance_hash(["in-b", "in-a"], "src-hash", "env-hash")
        == "7858572f099e516f8fabf133abbbf6cbd2ec3681c877b6decea0f95674918218"
    )


def test_display_subkey_is_pinned_to_a_known_value():
    """A plot's artifact id derives from its cell's hash; moving this orphans every figure."""
    cell_hash = compute_provenance_hash(["in-b", "in-a"], "src-hash", "env-hash")

    assert (
        derive_subkey(cell_hash, "__display__0")
        == "c30ab4b98c75d402bc93b4d18de59464dd011b0ba35965f8409f3bae3744c4bc"
    )


def test_provenance_hash_ordering_invariance():
    input_hashes1 = ["hash1", "hash2", "hash3"]
    input_hashes2 = ["hash3", "hash1", "hash2"]
    source_hash = compute_source_hash("x = 1")
    env_hash = compute_source_hash("env")

    hash1 = compute_provenance_hash(input_hashes1, source_hash, env_hash)
    hash2 = compute_provenance_hash(input_hashes2, source_hash, env_hash)

    assert hash1 == hash2


def test_provenance_hash_changes_with_source():
    input_hashes = ["hash1"]
    source1 = compute_source_hash("x = 1")
    source2 = compute_source_hash("x = 2")
    env_hash = compute_source_hash("env")

    hash1 = compute_provenance_hash(input_hashes, source1, env_hash)
    hash2 = compute_provenance_hash(input_hashes, source2, env_hash)

    assert hash1 != hash2


def test_provenance_hash_changes_with_env():
    input_hashes = ["hash1"]
    source_hash = compute_source_hash("x = 1")
    env1 = compute_source_hash("env1")
    env2 = compute_source_hash("env2")

    hash1 = compute_provenance_hash(input_hashes, source_hash, env1)
    hash2 = compute_provenance_hash(input_hashes, source_hash, env2)

    assert hash1 != hash2


def test_provenance_hash_changes_with_inputs():
    source_hash = compute_source_hash("x = 1")
    env_hash = compute_source_hash("env")

    hash1 = compute_provenance_hash(["hash1"], source_hash, env_hash)
    hash2 = compute_provenance_hash(["hash2"], source_hash, env_hash)

    assert hash1 != hash2


def test_provenance_hash_empty_inputs():
    source_hash = compute_source_hash("x = 1")
    env_hash = compute_source_hash("env")

    hash1 = compute_provenance_hash([], source_hash, env_hash)
    hash2 = compute_provenance_hash([], source_hash, env_hash)

    assert hash1 == hash2


# derive_subkey


def test_derive_subkey_matches_legacy_inline_form():
    """derive_subkey must equal ``sha256(f"{parent}:{label}")``; existing cache keys depend on
    it."""
    parent = "a" * 64
    expected = hashlib.sha256(f"{parent}:varname".encode()).hexdigest()
    assert derive_subkey(parent, "varname") == expected


def test_derive_subkey_multi_label_matches_legacy_inline_form():
    """Same byte identity for the loop shape ``f"{parent1}:{parent2}:iter={k}"``."""
    expected = hashlib.sha256(b"p1:p2:iter=3").hexdigest()
    assert derive_subkey("p1", "p2", "iter=3") == expected


def test_derive_subkey_stable():
    assert derive_subkey("parent", "x") == derive_subkey("parent", "x")


def test_derive_subkey_label_distinguishes_outputs():
    """Two output variables of one cell get distinct artifact keys."""
    parent = "common"
    assert derive_subkey(parent, "x") != derive_subkey(parent, "y")


def test_derive_subkey_parent_distinguishes_cells():
    """Two cells with the same output name get distinct artifact keys."""
    assert derive_subkey("cell_a", "result") != derive_subkey("cell_b", "result")


def test_derive_subkey_zero_labels_is_just_parent_hash():
    parent = "abc"
    assert derive_subkey(parent) == hashlib.sha256(parent.encode()).hexdigest()


def test_safe_filename_stem_is_case_collision_proof():
    from strata.notebook.provenance import safe_filename_stem

    # All-lowercase names (incl. the __display__N convention) are unchanged, so
    # their blob filenames stay stable across upgrades.
    assert safe_filename_stem("data") == "data"
    assert safe_filename_stem("__display__0") == "__display__0"
    # A name with uppercase gets a short hash suffix.
    up = safe_filename_stem("Data")
    assert up.startswith("Data-") and up != "Data"
    # Data vs data must not collide even under case folding (macOS/APFS,
    # Windows), so their blob files stay distinct.
    assert up != safe_filename_stem("data")
    assert up.lower() != safe_filename_stem("data").lower()
    # Two uppercase variants of one name also stay distinct.
    assert safe_filename_stem("Data") != safe_filename_stem("DATA")


def test_serializer_copy_matches_provenance_helper():
    """serializer._safe_filename_stem is a copy (the harness can't import strata): no drift."""
    from strata.notebook.provenance import safe_filename_stem as canonical
    from strata.notebook.serializer import _safe_filename_stem as copy

    for name in ["data", "Data", "Widget", "widget", "Tikhonov", "__display__0", "DF", "a_b", "X1"]:
        assert copy(name) == canonical(name), name
