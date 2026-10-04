"""Tests for environment hashing."""

import hashlib

from strata.notebook.env import (
    _runtime_uv_closure_fingerprint,
    collect_referenced_env_keys,
    compute_lockfile_hash,
    narrow_env_for_provenance,
)


def test_lockfile_hash_stability(tmp_path):
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("[[package]]\nname = 'pandas'\n")

    hash1 = compute_lockfile_hash(tmp_path)
    hash2 = compute_lockfile_hash(tmp_path)

    assert hash1 == hash2


def test_lockfile_hash_changes_with_content(tmp_path):
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


# --- dev-group exclusion from the provenance hash ---


def _uv_lock(
    *,
    runtime: dict[str, str],
    dev: dict[str, str] | None = None,
    transitive: dict[str, tuple[str, list[str]]] | None = None,
    project: str = "probe",
) -> str:
    """Build a minimal but realistic uv.lock for a notebook whose project is *project*.

    ``runtime`` / ``dev`` map name to version for the root's direct deps; ``transitive`` maps
    name to ``(version, [dep names])``. Each package gets an sdist hash from its name and
    version, so a version bump changes content.
    """
    dev = dev or {}
    transitive = transitive or {}
    lines = ["version = 1", "revision = 3", 'requires-python = ">=3.12"', ""]

    # Root project package.
    lines.append("[[package]]")
    lines.append(f'name = "{project}"')
    lines.append('version = "0.1.0"')
    lines.append('source = { virtual = "." }')
    lines.append("dependencies = [")
    for name in sorted(runtime):
        lines.append(f'    {{ name = "{name}" }},')
    lines.append("]")
    if dev:
        lines.append("")
        lines.append("[package.dev-dependencies]")
        lines.append("dev = [")
        for name in sorted(dev):
            lines.append(f'    {{ name = "{name}" }},')
        lines.append("]")
    lines.append("")

    # Resolved packages (runtime direct + dev direct + transitive).
    resolved: dict[str, tuple[str, list[str]]] = {}
    for name, version in {**runtime, **dev}.items():
        resolved[name] = (version, [])
    for name, (version, deps) in transitive.items():
        resolved[name] = (version, deps)

    for name in sorted(resolved):
        version, deps = resolved[name]
        lines.append("[[package]]")
        lines.append(f'name = "{name}"')
        lines.append(f'version = "{version}"')
        lines.append('source = { registry = "https://pypi.org/simple" }')
        digest = hashlib.sha256(f"{name}@{version}".encode()).hexdigest()
        lines.append(f'sdist = {{ url = "https://x/{name}.tar.gz", hash = "sha256:{digest}" }}')
        if deps:
            lines.append("dependencies = [")
            for dep in sorted(deps):
                lines.append(f'    {{ name = "{dep}" }},')
            lines.append("]")
        lines.append("")

    return "\n".join(lines)


def _hash_with(tmp_path, lock_text: str) -> str:
    (tmp_path / "uv.lock").write_text(lock_text)
    return compute_lockfile_hash(tmp_path)


def test_dev_dependency_does_not_change_hash(tmp_path):
    """Adding a dev tool (and bumping it) must not change the provenance hash."""
    base = _hash_with(tmp_path, _uv_lock(runtime={"cloudpickle": "3.1.2"}, dev={"pytest": "9.1.1"}))
    # Bump the dev tool's version and add a second dev tool: the runtime closure
    # is identical, so the hash must not move.
    bumped = _hash_with(
        tmp_path,
        _uv_lock(
            runtime={"cloudpickle": "3.1.2"},
            dev={"pytest": "9.9.9", "ruff": "0.14.0"},
        ),
    )
    assert base == bumped


def test_runtime_dependency_change_does_change_hash(tmp_path):
    """A runtime dep version bump must change the hash (cache correctness)."""
    before = _hash_with(
        tmp_path, _uv_lock(runtime={"cloudpickle": "3.1.2"}, dev={"pytest": "9.1.1"})
    )
    after = _hash_with(
        tmp_path, _uv_lock(runtime={"cloudpickle": "3.2.0"}, dev={"pytest": "9.1.1"})
    )
    assert before != after


def test_transitive_runtime_upgrade_changes_hash(tmp_path):
    """A transitive runtime dep upgrade (reached via the graph) changes the hash.

    Hashing only the root's direct deps would miss this and under-invalidate.
    """
    before = _hash_with(
        tmp_path,
        _uv_lock(
            runtime={"pandas": "2.0.0"},
            dev={"pytest": "9.1.1"},
            transitive={"pandas": ("2.0.0", ["numpy"]), "numpy": ("2.0.0", [])},
        ),
    )
    after = _hash_with(
        tmp_path,
        _uv_lock(
            runtime={"pandas": "2.0.0"},
            dev={"pytest": "9.1.1"},
            transitive={"pandas": ("2.0.0", ["numpy"]), "numpy": ("2.4.0", [])},
        ),
    )
    assert before != after


def test_dev_only_transitive_does_not_change_hash(tmp_path):
    """A package pulled in ONLY by a dev tool is excluded from the hash."""
    base = _hash_with(
        tmp_path,
        _uv_lock(
            runtime={"cloudpickle": "3.1.2"},
            dev={"pytest": "9.1.1"},
            transitive={"pytest": ("9.1.1", ["pluggy"]), "pluggy": ("1.6.0", [])},
        ),
    )
    # pluggy (dev-only transitive) upgrades; the hash must not move.
    bumped = _hash_with(
        tmp_path,
        _uv_lock(
            runtime={"cloudpickle": "3.1.2"},
            dev={"pytest": "9.1.1"},
            transitive={"pytest": ("9.1.1", ["pluggy"]), "pluggy": ("1.7.0", [])},
        ),
    )
    assert base == bumped


def test_notebooks_named_apart_with_the_same_dependencies_get_one_hash(tmp_path):
    """Two notebooks that differ only in their project name share an env hash, dev group or not."""
    for dev in (None, {"pytest": "9.1.1"}):
        nb0 = _uv_lock(runtime={"cloudpickle": "3.1.2"}, dev=dev, project="nb0")
        nb1 = _uv_lock(runtime={"cloudpickle": "3.1.2"}, dev=dev, project="nb1")
        assert nb0 != nb1
        assert _hash_with(tmp_path, nb0) == _hash_with(tmp_path, nb1)


def test_a_real_dependency_difference_still_changes_the_hash(tmp_path):
    """Without a dev group, a runtime version bump or an added dependency changes the hash."""
    base = _hash_with(tmp_path, _uv_lock(runtime={"cloudpickle": "3.1.2"}, project="nb0"))
    bumped = _hash_with(tmp_path, _uv_lock(runtime={"cloudpickle": "3.2.0"}, project="nb1"))
    added = _hash_with(
        tmp_path, _uv_lock(runtime={"cloudpickle": "3.1.2", "six": "1.17.0"}, project="nb1")
    )
    assert len({base, bumped, added}) == 3


def test_the_dev_group_path_folds_the_runtime_closure(tmp_path):
    """With a dev group, the uv part is the runtime-closure fingerprint, as before."""
    lock_text = _uv_lock(runtime={"cloudpickle": "3.1.2"}, dev={"pytest": "9.1.1"})
    fingerprint = _runtime_uv_closure_fingerprint(lock_text.encode())
    assert fingerprint is not None
    expected = hashlib.sha256(b"\0uv-runtime=" + fingerprint).hexdigest()
    assert _hash_with(tmp_path, lock_text) == expected


def test_unparseable_lock_falls_back_to_raw_bytes(tmp_path):
    """A malformed uv.lock does not crash; the hash folds its raw bytes."""
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text("this is = not [valid toml")
    actual = compute_lockfile_hash(tmp_path)
    expected = hashlib.sha256(lockfile.read_bytes()).hexdigest()
    assert actual == expected


def test_lockfile_hash_renv_lock_changes_digest(tmp_path):
    """Changing only renv.lock changes the hash, invalidating every R cell."""
    (tmp_path / "uv.lock").write_text("[[package]]\nname = 'pandas'\n")
    (tmp_path / "renv.lock").write_text('{"Packages": {"arrow": "1.0"}}')
    before = compute_lockfile_hash(tmp_path)

    (tmp_path / "renv.lock").write_text('{"Packages": {"arrow": "2.0"}}')
    after = compute_lockfile_hash(tmp_path)

    assert before != after, "renv.lock edit must invalidate the lockfile hash"


def test_lockfile_hash_renv_only_notebook(tmp_path):
    """An R-only notebook (renv.lock, no uv.lock) hashes without crashing, and stably."""
    (tmp_path / "renv.lock").write_text('{"R": {"Version": "4.4.0"}}')

    a = compute_lockfile_hash(tmp_path)
    b = compute_lockfile_hash(tmp_path)
    assert a == b

    # Sanity: the renv tag prefix makes the R-only hash distinct
    # from the no-lockfiles sentinel (empty sha256).
    assert a != hashlib.sha256(b"").hexdigest()


def test_lockfile_hash_uv_and_renv_combined(tmp_path):
    """Adding renv.lock to a uv-only notebook changes the hash too."""
    (tmp_path / "uv.lock").write_text("[[package]]\nname = 'pandas'\n")
    uv_only = compute_lockfile_hash(tmp_path)

    (tmp_path / "renv.lock").write_text('{"Packages": {"arrow": "1.0"}}')
    uv_plus_renv = compute_lockfile_hash(tmp_path)

    assert uv_only != uv_plus_renv


def test_collect_referenced_env_keys_subscript():
    assert collect_referenced_env_keys("import os\nx = os.environ['APP_MODE']") == {"APP_MODE"}


def test_collect_referenced_env_keys_get_and_getenv():
    source = "import os\na = os.environ.get('A', 'default')\nb = os.getenv('B')\n"
    assert collect_referenced_env_keys(source) == {"A", "B"}


def test_collect_referenced_env_keys_from_os_import_aliases():
    source = (
        "from os import environ, getenv\nx = environ['A']\ny = environ.get('B')\nz = getenv('C')\n"
    )
    assert collect_referenced_env_keys(source) == {"A", "B", "C"}


def test_collect_referenced_env_keys_ignores_dynamic_lookup():
    """Non-literal keys cannot be resolved statically, so they are ignored."""
    source = "import os\nkey = 'A'\nx = os.environ[key]\n"
    assert collect_referenced_env_keys(source) == set()


def test_collect_referenced_env_keys_syntax_error_returns_empty():
    assert collect_referenced_env_keys("def broken(:") == set()


def test_narrow_env_for_provenance_drops_unreferenced_keys():
    """Notebook-level env vars that a cell does not reference are dropped."""
    source = "import os\nx = os.environ['USED']"
    resolved = {"USED": "1", "UNUSED": "secret", "OPENAI_API_KEY": "sk"}

    narrowed = narrow_env_for_provenance(source, resolved)

    assert narrowed == {"USED": "1"}


def test_narrow_env_for_provenance_keeps_declared_keys():
    """Declared keys (annotations or persisted overrides) are kept even when the cell never
    reads them; the declaration is the opt-in.
    """
    source = "x = 1"  # no references
    resolved = {"DECLARED": "hello", "AMBIENT": "ignored"}

    narrowed = narrow_env_for_provenance(source, resolved, declared_keys={"DECLARED"})

    assert narrowed == {"DECLARED": "hello"}
