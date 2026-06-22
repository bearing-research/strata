"""Tests for environment hashing."""

import hashlib

from strata.notebook.env import (
    collect_referenced_env_keys,
    compute_lockfile_hash,
    narrow_env_for_provenance,
)


def test_lockfile_hash_stability(tmp_path):
    """Same lockfile should produce same hash."""
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("[[package]]\nname = 'pandas'\n")

    hash1 = compute_lockfile_hash(tmp_path)
    hash2 = compute_lockfile_hash(tmp_path)

    assert hash1 == hash2


def test_lockfile_hash_changes_with_content(tmp_path):
    """Different lockfile content should produce different hash."""
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("[[package]]\nname = 'pandas'\n")
    hash1 = compute_lockfile_hash(tmp_path)

    lockfile.write_text("[[package]]\nname = 'numpy'\n")
    hash2 = compute_lockfile_hash(tmp_path)

    assert hash1 != hash2


def test_lockfile_hash_missing_lockfile(tmp_path):
    """Missing lockfile should return sentinel hash."""
    hash_val = compute_lockfile_hash(tmp_path)

    # Sentinel hash is sha256 of empty string
    expected = hashlib.sha256(b"").hexdigest()

    assert hash_val == expected


def test_lockfile_hash_unchanged_for_uv_only_notebook(tmp_path):
    """Adding renv.lock support must not invalidate uv-only notebooks.

    Regression for #59 PR 4: ``compute_lockfile_hash`` was extended
    to fold ``renv.lock`` into the digest, but a Python-only
    notebook (no renv.lock present) must still produce the same
    bytes-as-input hash it did pre-change — otherwise every cached
    R-free notebook on disk loses its cache the moment this lands.
    """
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("[[package]]\nname = 'pandas'\nversion = '2.0'\n")

    actual = compute_lockfile_hash(tmp_path)
    expected = hashlib.sha256(lockfile.read_bytes()).hexdigest()

    assert actual == expected, (
        "uv-only notebook hash must match raw sha256(uv.lock) for back-compat with pre-#59 caches."
    )


def test_lockfile_hash_renv_lock_changes_digest(tmp_path):
    """A renv.lock change must produce a different hash.

    Acceptance criterion from #59: ``renv.lock change invalidates
    all R cells (env hash changed)``. The notebook here has both
    uv.lock and renv.lock; we mutate only renv.lock and assert the
    digest drifts.
    """
    (tmp_path / "uv.lock").write_text("[[package]]\nname = 'pandas'\n")
    (tmp_path / "renv.lock").write_text('{"Packages": {"arrow": "1.0"}}')
    before = compute_lockfile_hash(tmp_path)

    (tmp_path / "renv.lock").write_text('{"Packages": {"arrow": "2.0"}}')
    after = compute_lockfile_hash(tmp_path)

    assert before != after, "renv.lock edit must invalidate the lockfile hash"


def test_lockfile_hash_renv_only_notebook(tmp_path):
    """R-only notebook (no uv.lock, just renv.lock) produces a stable hash.

    Future configuration — there's no concrete user story for an
    R-only Strata notebook yet, but the helper must not crash, and
    repeated calls with the same renv.lock must agree.
    """
    (tmp_path / "renv.lock").write_text('{"R": {"Version": "4.4.0"}}')

    a = compute_lockfile_hash(tmp_path)
    b = compute_lockfile_hash(tmp_path)
    assert a == b

    # Sanity: the renv tag prefix makes the R-only hash distinct
    # from the no-lockfiles sentinel (empty sha256).
    assert a != hashlib.sha256(b"").hexdigest()


def test_lockfile_hash_uv_and_renv_combined(tmp_path):
    """Adding renv.lock to an existing uv.lock notebook drifts the hash.

    Pins the rule "renv.lock contributes to the hash" from the
    *other* direction: not just renv→renv edits, but introducing
    renv.lock into a previously uv-only notebook also invalidates.
    """
    (tmp_path / "uv.lock").write_text("[[package]]\nname = 'pandas'\n")
    uv_only = compute_lockfile_hash(tmp_path)

    (tmp_path / "renv.lock").write_text('{"Packages": {"arrow": "1.0"}}')
    uv_plus_renv = compute_lockfile_hash(tmp_path)

    assert uv_only != uv_plus_renv


def test_collect_referenced_env_keys_subscript():
    """``os.environ['KEY']`` should be detected."""
    assert collect_referenced_env_keys("import os\nx = os.environ['APP_MODE']") == {"APP_MODE"}


def test_collect_referenced_env_keys_get_and_getenv():
    """``os.environ.get`` and ``os.getenv`` literal keys are detected."""
    source = "import os\na = os.environ.get('A', 'default')\nb = os.getenv('B')\n"
    assert collect_referenced_env_keys(source) == {"A", "B"}


def test_collect_referenced_env_keys_from_os_import_aliases():
    """``from os import environ, getenv`` usages are detected."""
    source = (
        "from os import environ, getenv\nx = environ['A']\ny = environ.get('B')\nz = getenv('C')\n"
    )
    assert collect_referenced_env_keys(source) == {"A", "B", "C"}


def test_collect_referenced_env_keys_ignores_dynamic_lookup():
    """Non-literal keys are ignored — they cannot be statically resolved."""
    source = "import os\nkey = 'A'\nx = os.environ[key]\n"
    assert collect_referenced_env_keys(source) == set()


def test_collect_referenced_env_keys_syntax_error_returns_empty():
    """Invalid source must not crash; return an empty set."""
    assert collect_referenced_env_keys("def broken(:") == set()


def test_narrow_env_for_provenance_drops_unreferenced_keys():
    """Notebook-level env vars that a cell does not reference are dropped."""
    source = "import os\nx = os.environ['USED']"
    resolved = {"USED": "1", "UNUSED": "secret", "OPENAI_API_KEY": "sk"}

    narrowed = narrow_env_for_provenance(source, resolved)

    assert narrowed == {"USED": "1"}


def test_narrow_env_for_provenance_keeps_declared_keys():
    """Explicitly declared keys (annotations or persisted overrides) are kept
    even when the cell body never reads them — the declaration is the
    explicit opt-in signal."""
    source = "x = 1"  # no references
    resolved = {"DECLARED": "hello", "AMBIENT": "ignored"}

    narrowed = narrow_env_for_provenance(source, resolved, declared_keys={"DECLARED"})

    assert narrowed == {"DECLARED": "hello"}
