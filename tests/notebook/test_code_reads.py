"""What a scratchpad cell's run read, as ``code_reads`` records it (design 4.3).

Each case runs its body in a fresh interpreter: the recorder patches the import system and
installs an audit hook that cannot be removed.
"""

from __future__ import annotations

import hashlib
import json
import marshal
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from strata.notebook import code_reads
from strata.notebook.code_reads import ABSENT

_RUN = """\
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("_code_reads", {module!r})
code_reads = importlib.util.module_from_spec(spec)
spec.loader.exec_module(code_reads)
with code_reads.recording({exclude!r}, pycache={pycache!r}) as reads:
{body}
record = reads.finish()
print(json.dumps({{"record": record, "value": globals().get("value")}}))
"""


def _record(
    tmp_path: Path, body: str, *, exclude: list[str] | None = None, pycache: str | None = None
) -> dict:
    script = _RUN.format(
        module=code_reads.__file__,
        exclude=exclude or [],
        pycache=pycache,
        body=textwrap.indent(textwrap.dedent(body), "    "),
    )
    run = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "0"},
    )
    return json.loads(run.stdout.splitlines()[-1])


def _sha(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_module_is_recorded_by_the_bytes_that_ran(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "geom.py").write_text("SIDE = 3\n")

    got = _record(
        tmp_path,
        f"""
        sys.path.insert(0, {str(lib)!r})
        import geom
        value = geom.SIDE
        """,
    )

    reads = got["record"]["reads"]
    assert got["value"] == 3
    assert reads[str(lib / "geom.py")] == _sha(lib / "geom.py")
    # What the path finder looked for first, and did not find.
    assert reads[str(lib / "geom")] == ABSENT
    assert all(reads[str(lib / f"geom{s}")] == ABSENT for s in code_reads._SUFFIXES[:-2])
    assert got["record"]["unstable"] is None


def test_a_module_runs_from_its_source_never_a_stale_pyc(tmp_path):
    # A same-size edit with its mtime put back fools the .pyc check, not the recorder.
    lib = tmp_path / "lib"
    lib.mkdir()
    module = lib / "geom.py"
    module.write_text("SIDE = 3\n")
    subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(lib)!r}); import geom"],
        check=True,
    )
    assert list((lib / "__pycache__").iterdir())
    before = module.stat()
    module.write_text("SIDE = 4\n")
    os.utime(module, ns=(before.st_atime_ns, before.st_mtime_ns))

    got = _record(tmp_path, f"sys.path.insert(0, {str(lib)!r})\nimport geom\nvalue = geom.SIDE")

    assert got["value"] == 4
    assert got["record"]["reads"][str(module)] == _sha(module)


def test_a_module_loaded_from_its_file_is_recorded(tmp_path):
    module = tmp_path / "loose.py"
    module.write_text("value = 7\n")

    got = _record(
        tmp_path,
        f"""
        spec = importlib.util.spec_from_file_location("loose", {str(module)!r})
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        value = loaded.value
        """,
    )

    assert got["value"] == 7
    assert got["record"]["reads"][str(module)] == _sha(module)


def test_an_import_that_fell_back_records_where_it_looked(tmp_path):
    # A module that appears later would change what the cell does with every read the same.
    got = _record(
        tmp_path,
        f"""
        sys.path.insert(0, {str(tmp_path)!r})
        try:
            import fastpath
            value = 1
        except ImportError:
            value = 2
        """,
    )

    assert got["value"] == 2
    assert got["record"]["reads"][str(tmp_path / "fastpath.py")] == ABSENT


def test_files_opened_and_listed_are_recorded_and_its_own_output_is_not(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.csv").write_text("a,b\n")
    (tmp_path / "data" / ".hidden").write_text("")

    got = _record(
        tmp_path,
        """
        import glob, os
        value = open("data/x.csv").read()
        # Opening a directory reads nothing of it; listing it does.
        os.close(os.open("data", os.O_RDONLY))
        names = sorted(os.listdir("data"))
        open("out.txt", "w").write(value)
        value += open("out.txt").read()
        try:
            open("missing.csv")
        except FileNotFoundError:
            pass
        """,
    )

    reads = got["record"]["reads"]
    assert reads[str(tmp_path / "data" / "x.csv")] == _sha(tmp_path / "data" / "x.csv")
    assert (
        reads[str(tmp_path / "data")] == code_reads.observe(str(tmp_path / "data"), listing=True)[0]
    )
    assert reads[str(tmp_path / "missing.csv")] == ABSENT
    assert str(tmp_path / "out.txt") not in reads
    assert set(got["record"]["stat"]) <= set(reads)


def test_the_environment_and_what_is_excluded_are_not_recorded(tmp_path):
    (tmp_path / "store").mkdir()
    (tmp_path / "store" / "kept.bin").write_text("x")

    got = _record(
        tmp_path,
        """
        import decimal
        value = open("store/kept.bin").read()
        """,
        exclude=[str(tmp_path / "store")],
    )

    reads = got["record"]["reads"]
    # Only where the import looked before the stdlib, as the start directory (first on the
    # path under `python -c`, where a decimal.py would shadow it): no file was read.
    assert str(tmp_path / "decimal.py") in reads
    assert [path for path, seen in reads.items() if seen != ABSENT] == []


def test_a_file_changed_while_the_cell_ran_makes_it_unstable(tmp_path):
    data = tmp_path / "x.csv"
    data.write_text("1\n")

    got = _record(
        tmp_path,
        f"""
        import subprocess
        value = open("x.csv").read()
        subprocess.run(["{sys.executable}", "-c", "open({str(data)!r}, 'a').write('2')"])
        """,
    )

    assert got["record"]["unstable"] == f"{data} changed while the cell ran"


def test_a_file_the_cell_read_then_wrote_makes_it_unstable(tmp_path):
    (tmp_path / "x.csv").write_text("1\n")

    got = _record(tmp_path, 'value = open("x.csv").read()\nopen("x.csv", "a").write("2")')

    assert got["record"]["unstable"] == f"the cell wrote {tmp_path / 'x.csv'}, which it had read"


def test_an_import_hook_the_cell_adds_makes_it_unstable(tmp_path):
    got = _record(
        tmp_path,
        """
        class Finder:
            def find_spec(self, name, path, target=None):
                return None
        sys.meta_path.append(Finder())
        import email.mime.text
        """,
    )

    assert got["record"]["unstable"] == "the cell changed sys.meta_path (__main__.Finder)"


def test_a_stat_fingerprint_is_racy_while_young(tmp_path):
    path = tmp_path / "f"
    path.write_text("x")
    now = code_reads.fingerprint(str(path), __import__("time").time_ns())
    assert now is not None
    assert code_reads.racy(now)
    old = [*now[:5], now[5] + 10 * code_reads.RACY_NS]
    assert not code_reads.racy(old)
    assert code_reads.racy([*old[:4], 0, old[5]])


def test_a_directory_the_cell_made_and_removed_is_its_own(tmp_path):
    got = _record(
        tmp_path,
        """
        import os, tempfile
        with tempfile.TemporaryDirectory(dir=".") as d:
            open(f"{d}/x", "w").write("1")
            value = open(f"{d}/x").read() + str(len(os.listdir(d)))
        """,
    )

    assert got["value"] == "11"
    assert got["record"]["unstable"] is None
    # Only where its imports looked: nothing of the directory or the file in it.
    assert all(seen == ABSENT for seen in got["record"]["reads"].values())


def test_the_runs_own_directory_is_not_recorded(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "input.arrow").write_text("x")
    (tmp_path / "data.txt").write_text("y")
    script = f"""
import contextlib, importlib.util, json, types
spec = importlib.util.spec_from_file_location("_code_reads", {code_reads.__file__!r})
code_reads = importlib.util.module_from_spec(spec)
spec.loader.exec_module(code_reads)
observed = types.SimpleNamespace(
    on_native_read=None, on_native_write=None, quiet=False, hushed=contextlib.nullcontext
)
manifest = {{"code_reads": {{"exclude": []}}, "output_dir": {str(run)!r}}}
with code_reads.for_run(manifest) as reads, code_reads.observing(reads, observed):
    told = observed.on_native_read == reads.file
    open({str(run / "input.arrow")!r}).read()
    open({str(tmp_path / "data.txt")!r}).read()
after = observed.on_native_read
print(json.dumps({{"told": told, "after": after, "reads": reads.finish()["reads"]}}))
"""
    run_ = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    got = json.loads(run_.stdout.splitlines()[-1])

    assert list(got["reads"]) == [str(tmp_path / "data.txt")]
    # A native reader's files are told to the record while it runs, and only then.
    assert got["told"] and got["after"] is None


def test_a_large_file_has_one_form_however_soon_it_is_observed(tmp_path, monkeypatch):
    monkeypatch.setattr(code_reads, "LARGE_BYTES", 4)
    path = tmp_path / "big.bin"
    path.write_bytes(b"0123456789")

    seen, print_ = code_reads.observe(str(path))

    assert print_ is not None and code_reads.racy(print_)
    assert seen == "stat:" + ":".join(map(str, print_[:5]))


def test_the_end_of_a_run_reads_again_only_what_moved(tmp_path, monkeypatch):
    path = tmp_path / "data.txt"
    path.write_text("x")
    # No stat is too young to trust here.
    monkeypatch.setattr(code_reads, "RACY_NS", -(10**18))
    reads = code_reads.Reads([])
    reads._hooks = (list(sys.meta_path), list(sys.path_hooks))
    reads.file(str(path))
    observed: list[str] = []
    real = code_reads.observe
    monkeypatch.setattr(
        code_reads, "observe", lambda p, **kw: (observed.append(p), real(p, **kw))[1]
    )

    assert reads.finish()["unstable"] is None
    assert observed == []

    path.write_text("yy")
    assert reads.finish()["unstable"] == f"{path} changed while the cell ran"
    assert observed == [str(path)]


def _script(tmp_path: Path, script: str) -> dict:
    run = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(run.stdout.splitlines()[-1])


def test_what_a_native_writer_wrote_is_the_runs_own(tmp_path):
    # The read audit lists the target before and after the write: its work, not the cell's.
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "keep.txt").write_text("k")

    got = _script(
        tmp_path,
        """
        import json
        import pyarrow as pa, pyarrow.parquet as pq
        from strata.notebook import code_reads, read_audit
        manifest = {"code_reads": {"exclude": []}}
        run, audit = code_reads.for_run(manifest), read_audit.recording()
        with run as reads, audit as observed, code_reads.observing(reads, observed):
            pq.write_to_dataset(pa.table({"x": [1, 2]}), root_path="out")
            pq.write_table(pa.table({"x": [3]}), "out/part.parquet")
            # Read back, what it wrote is still its own.
            pq.read_table("out/part.parquet")
        print(json.dumps(reads.finish()))
        """,
    )

    assert got["unstable"] is None
    assert not [path for path in got["reads"] if "out" in Path(path).parts]


def test_the_runs_own_folders_are_not_recorded(tmp_path):
    # A loop's ``# @output_dir`` folders live outside its run directory.
    own, data = tmp_path / "own", tmp_path / "data"
    for folder in (own, data):
        folder.mkdir()
        (folder / "f.txt").write_text("x")
    mounts = {
        "ckpt": {"uri": "", "mode": "rw", "local_path": str(own)},
        "data": {"uri": data.as_uri(), "mode": "ro", "local_path": str(data)},
    }

    got = _script(
        tmp_path,
        f"""
        import contextlib, json, types
        from strata.notebook import code_reads
        observed = types.SimpleNamespace(
            on_native_read=None, on_native_write=None, quiet=False, hushed=contextlib.nullcontext
        )
        manifest = {{"code_reads": {{"exclude": []}}, "mounts": {mounts!r}}}
        with code_reads.for_run(manifest) as reads, code_reads.observing(reads, observed):
            open({str(own / "f.txt")!r}).read()
            open({str(data / "f.txt")!r}).read()
        print(json.dumps(reads.finish()))
        """,
    )

    assert list(got["reads"]) == [str(data / "f.txt")]


def test_an_import_from_a_removed_directory_still_imports(tmp_path):
    gone = tmp_path / "gone"
    gone.mkdir()

    got = _record(
        tmp_path,
        f"""
        import os
        sys.path.insert(0, "")
        os.chdir({str(gone)!r})
        os.rmdir({str(gone)!r})
        import colorsys
        value = colorsys.__name__
        """,
    )

    assert got["value"] == "colorsys"
    assert got["record"]["unstable"].startswith("the record failed: FileNotFoundError")


def test_a_record_that_fails_at_the_end_is_unstable_not_an_error(tmp_path):
    (tmp_path / "x.csv").write_text("1")

    got = _record(
        tmp_path,
        """
        value = open("x.csv").read()
        code_reads.fingerprint = lambda *_args: 1 / 0
        """,
    )

    assert got["record"] == {
        "reads": {},
        "stat": {},
        "unstable": "the record failed: ZeroDivisionError('division by zero')",
    }


def test_a_path_hook_the_environment_installed_is_its_own(tmp_path):
    # As setuptools' strict editable install puts on sys.path.
    env = tmp_path / "env"
    env.mkdir()
    (env / "hookmod.py").write_text(
        "class Finder:\n"
        "    def __init__(self, path):\n"
        "        self.path = path\n"
        "    def find_spec(self, name, target=None):\n"
        "        return None\n"
    )

    got = _script(
        tmp_path,
        f"""
        import json, sys
        sys.path.insert(0, {str(env)!r})
        import hookmod
        sys.path.insert(0, "x.__path_hook__")
        sys.path_importer_cache["x.__path_hook__"] = hookmod.Finder("x.__path_hook__")
        from strata.notebook import code_reads
        with code_reads.recording([{str(env)!r}]) as reads:
            import colorsys
        print(json.dumps(reads.finish()))
        """,
    )

    assert got["unstable"] is None


def test_an_import_only_untracked_directories_serve_searches_nothing(tmp_path):
    # A submodule of an environment package: its package's __path__ is all it searches.
    got = _record(
        tmp_path,
        """
        searched = []
        search = code_reads._Absences._search
        code_reads._Absences._search = lambda self, name, *a: (
            searched.append(name) or search(self, name, *a)
        )
        import email.mime.audio
        value = searched
        """,
    )

    assert got["value"]
    assert [name for name in got["value"] if "." in name] == []


def test_a_package_with_an_empty_path_searches_nowhere(tmp_path):
    # As six's ``six.moves``: a submodule of it is found by a hook, never on sys.path.
    got = _record(
        tmp_path,
        """
        import types
        pseudo = types.ModuleType("pseudo")
        pseudo.__path__ = []
        sys.modules["pseudo"] = pseudo
        try:
            import pseudo.child
        except ImportError:
            pass
        """,
    )

    assert not [path for path in got["record"]["reads"] if "child" in path]


_FOR_RUN = """\
import contextlib, importlib.util, json, sys, types
spec = importlib.util.spec_from_file_location("_code_reads", {module!r})
code_reads = importlib.util.module_from_spec(spec)
spec.loader.exec_module(code_reads)
manifest = {{"code_reads": {{"exclude": [], "anchor": {anchor!r}}}}}
observed = types.SimpleNamespace(quiet=False, hushed=contextlib.nullcontext)
with code_reads.for_run(manifest) as reads, code_reads.observing(reads, observed):
    before = {anchor!r} in sys.path
    with code_reads.anchored(reads):
        during = sys.path[0]
print(json.dumps({{"before": before, "during": during, "after": {anchor!r} in sys.path}}))
"""


def test_the_project_is_on_the_path_for_the_cell_alone(tmp_path):
    """What the harness imports itself, loading inputs or serializing outputs, never finds
    the project's."""
    script = _FOR_RUN.format(module=code_reads.__file__, anchor=str(tmp_path))
    run = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    got = json.loads(run.stdout.splitlines()[-1])
    assert got == {"before": False, "during": str(tmp_path), "after": False}


_ANCHORED = """\
import contextlib, importlib.util, json, sys, types
spec = importlib.util.spec_from_file_location("_code_reads", {module!r})
code_reads = importlib.util.module_from_spec(spec)
spec.loader.exec_module(code_reads)
anchor = {anchor!r}
sys.path.append("".join(list(anchor)))  # the same folder, as PYTHONPATH lists it
manifest = {{"code_reads": {{"exclude": [], "anchor": anchor}}}}
with code_reads.for_run(manifest) as reads:
    with code_reads.anchored(reads):
        import anchored_mod
        sys.path.remove(anchor)  # the cell takes the folder off the path
    kept = sys.path.count(anchor)
    with code_reads.anchored(reads):
        # The cell rebuilds the path, from equal strings: no entry is the inserted one.
        sys.path[:] = ["".join(list(entry)) for entry in sys.path]
    rebuilt = sys.path.count(anchor)
imported = "anchored_mod" in reads.imported
print(json.dumps({{"imported": imported, "kept": kept, "rebuilt": rebuilt}}))
"""


def test_anchored_notes_what_it_imported_and_removes_only_its_own_entry(tmp_path):
    """What it imported is a value's own code to the audit. Its entry goes, not the
    folder's other one (PYTHONPATH's), which a warm worker keeps for its next run."""
    (tmp_path / "anchored_mod.py").write_text("VALUE = 1\n")
    script = _ANCHORED.format(module=code_reads.__file__, anchor=str(tmp_path))
    run = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path / "..", capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout.splitlines()[-1]) == {"imported": True, "kept": 1, "rebuilt": 1}


def test_the_probe_watches_the_interpreter_and_what_rewrites_the_path(tmp_path, monkeypatch):
    """An interpreter upgraded in place, a ``sitecustomize`` edited, a ``pyvenv.cfg``."""
    probed = code_reads.probe_environment(str(tmp_path), [])
    observed = probed["observed"]
    assert observed[os.path.realpath(sys.executable)][0] is None
    assert observed[os.path.realpath(os.__file__)][0] is None
    assert observed[os.path.join(sys.prefix, "pyvenv.cfg")][0] is False
    entry = next(p for p in sys.path if p and os.path.isdir(p))
    assert observed[os.path.join(entry, "sitecustomize.py")][0] is False


def test_a_folder_on_pythonpath_is_the_users_code_not_an_install_root(tmp_path, monkeypatch):
    """`pip install -t vendor` on PYTHONPATH: its modules are recorded as a run's reads."""
    vendor, installed = tmp_path / "vendor", tmp_path / "installed"
    for folder in (vendor, installed):
        (folder / "x-1.0.dist-info").mkdir(parents=True)
        monkeypatch.syspath_prepend(str(folder))
    monkeypatch.setenv("PYTHONPATH", str(vendor))
    roots = code_reads.probe_environment(str(tmp_path / "anchor"), [])["roots"]
    assert str(vendor) not in roots
    assert str(installed) in roots


# Imports one module, saying which modules were compiled from source.
_COUNTED = """
import builtins
real = builtins.compile
compiled = []
def counted(source, filename, *args, **kwargs):
    if str(filename).startswith({lib!r}):
        compiled.append(filename)
    return real(source, filename, *args, **kwargs)
builtins.compile = counted
sys.path.insert(0, {lib!r})
import {name}
value = [{name}.VALUE, compiled]
"""


def _cached(tmp_path: Path, name: str = "geom", pycache: Path | None = None) -> list:
    body = _COUNTED.format(lib=str(tmp_path / "lib"), name=name)
    return _record(tmp_path, body, pycache=str(pycache or tmp_path / "pycache"))["value"]


def _module(tmp_path: Path, name: str, value: str) -> Path:
    module = tmp_path / "lib" / f"{name}.py"
    module.parent.mkdir(exist_ok=True)
    module.write_text(f"VALUE = {value!r}\n")
    return module


def _entry(tmp_path: Path, module: Path) -> Path:
    cache = code_reads.CodeCache(str(tmp_path / "pycache"))
    return Path(cache.entry(hashlib.sha256(module.read_bytes()).hexdigest(), str(module)))


def test_a_module_is_compiled_once_per_source(tmp_path):
    module = _module(tmp_path, "geom", "first")

    assert _cached(tmp_path) == ["first", [str(module)]]
    assert _cached(tmp_path) == ["first", []]
    module.write_text('VALUE = "second"\n')
    assert _cached(tmp_path) == ["second", [str(module)]]


@pytest.mark.parametrize("damage", ["constant", "truncated", "version", "another_key"])
def test_a_damaged_entry_is_compiled_again_never_run(tmp_path, damage):
    module = _module(tmp_path, "geom", "first")
    other = _module(tmp_path, "other", "other")
    _cached(tmp_path)
    _cached(tmp_path, "other")
    entry = _entry(tmp_path, module)
    blob = entry.read_bytes()
    if damage == "constant":
        assert blob.count(b"first") == 1
        changed = blob.replace(b"first", b"fxrst")
        # Still code, computing something else: only the checksum stands in the way.
        header = code_reads.CodeCache("").header(
            hashlib.sha256(module.read_bytes()).hexdigest(), str(module)
        )
        fixed = len(code_reads._PYC_MAGIC) + code_reads._PYC_LENGTHS.size + len(header)
        namespace: dict = {}
        exec(marshal.loads(changed[fixed : -code_reads._PYC_DIGEST]), namespace)
        assert namespace["VALUE"] == "fxrst"
    elif damage == "truncated":
        changed = blob[: len(blob) // 2]
    elif damage == "version":
        changed = blob.replace(code_reads._PYC_MAGIC, b"STRATAPYC\x02", 1)
        changed = changed[: -code_reads._PYC_DIGEST]
        changed += hashlib.sha256(changed).digest()
    else:
        changed = _entry(tmp_path, other).read_bytes()
    entry.write_bytes(changed)

    assert _cached(tmp_path) == ["first", [str(module)]]
    assert _cached(tmp_path) == ["first", []]


def test_the_same_source_at_two_paths_keeps_its_own_filename(tmp_path):
    """The path is compiled in (``co_filename``, so tracebacks): it is part of the key."""
    source = "import sys\nVALUE = sys._getframe().f_code.co_filename\n"
    (tmp_path / "lib").mkdir()
    for name in ("one", "two"):
        (tmp_path / "lib" / f"{name}.py").write_text(source)
    _cached(tmp_path, "one")
    assert _cached(tmp_path, "two") == [
        str(tmp_path / "lib" / "two.py"),
        [str(tmp_path / "lib" / "two.py")],
    ]
    assert _cached(tmp_path, "one") == [str(tmp_path / "lib" / "one.py"), []]


def test_the_cache_keeps_the_most_recently_used_within_its_limit(tmp_path):
    cache = code_reads.CodeCache(str(tmp_path))
    compiled: list[str] = []

    def load(name: str) -> None:
        data = f"VALUE = {name!r}\n".encode()
        digest = hashlib.sha256(data).hexdigest()
        if cache.load(digest, name) is None:
            compiled.append(name)
            cache.store(digest, name, compile(data, name, "exec"))

    load("a")
    (size,) = {p.stat().st_size for p in tmp_path.iterdir()}
    cache.limit = int(size * 2.5)
    load("b")
    os.utime(cache.entry(hashlib.sha256(b"VALUE = 'a'\n").hexdigest(), "a"), ns=(1, 1))
    os.utime(cache.entry(hashlib.sha256(b"VALUE = 'b'\n").hexdigest(), "b"), ns=(2, 2))
    load("a")  # used again: now the most recent
    load("c")
    assert len(list(tmp_path.iterdir())) == 2
    compiled.clear()
    load("a")
    load("c")
    load("b")
    assert compiled == ["b"]


def test_a_cache_that_cannot_be_written_compiles_every_run(tmp_path):
    module = _module(tmp_path, "geom", "first")
    (tmp_path / "file").write_text("")
    pycache = tmp_path / "file" / "pycache"

    assert _cached(tmp_path, pycache=pycache) == ["first", [str(module)]]
    assert _cached(tmp_path, pycache=pycache) == ["first", [str(module)]]


def test_the_read_audit_sees_nothing_of_the_cache(tmp_path):
    """Its files are the recorder's, never an undeclared read or write of the cell's."""
    _module(tmp_path, "geom", "first")
    script = f"""
    import json, sys
    from strata.notebook import code_reads, read_audit
    manifest = {{"code_reads": {{"exclude": [], "pycache": {str(tmp_path / "pycache")!r}}}}}
    run = code_reads.for_run(manifest)
    with run as reads, read_audit.recording() as observed, code_reads.observing(reads, observed):
        sys.path.insert(0, {str(tmp_path / "lib")!r})
        import geom
    print(json.dumps(sorted(observed.files)))
    """
    for _ in range(2):  # a miss, which stores, then a hit, which reads
        files = _script(tmp_path, script)
        assert str(tmp_path / "lib" / "geom.py") in files
        assert not [f for f in files if f.startswith(str(tmp_path / "pycache"))], files


def test_a_loader_compiling_code_of_its_own_is_never_cached(tmp_path):
    """An instrumenting import hook's code is not the plain code, either way round."""
    module = _module(tmp_path, "geom", "first")
    instrumented = f"""
    import importlib.machinery as machinery, importlib.util
    class Instrumenting(machinery.SourceFileLoader):
        def source_to_code(self, data, path, *args, **kwargs):
            data += b"\\nINSTRUMENTED = True\\n"
            return super().source_to_code(data, path, *args, **kwargs)
    path = {str(module)!r}
    spec = importlib.util.spec_from_file_location("geom", path, loader=Instrumenting("geom", path))
    geom = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geom)
    value = hasattr(geom, "INSTRUMENTED")
    """
    lib = str(tmp_path / "lib")
    plain = f"sys.path.insert(0, {lib!r})\nimport geom\nvalue = hasattr(geom, 'INSTRUMENTED')"
    pycache = str(tmp_path / "pycache")

    assert _record(tmp_path, plain, pycache=pycache)["value"] is False
    assert _record(tmp_path, instrumented, pycache=pycache)["value"] is True
    assert _record(tmp_path, plain, pycache=pycache)["value"] is False
    assert len(list((tmp_path / "pycache").iterdir())) == 1


def test_a_writer_killed_midway_leaves_nothing_that_counts(tmp_path):
    cache = code_reads.CodeCache(str(tmp_path))
    stale = tmp_path / ".tmp-stale"
    stale.write_bytes(b"x" * 1000)
    os.utime(stale, ns=(1, 1))
    live = tmp_path / ".tmp-live"
    live.write_bytes(b"x" * 1000)
    data = b"VALUE = 1\n"
    cache.store(hashlib.sha256(data).hexdigest(), "m", compile(data, "m", "exec"))

    assert not stale.exists()
    assert live.exists()  # another writer's, still being written
    assert cache._total == Path(cache.entry(hashlib.sha256(data).hexdigest(), "m")).stat().st_size


def test_a_compiler_swapped_on_one_loader_is_never_cached(tmp_path):
    """Installed on the instance, not the class: still not the stdlib's compiler."""
    module = _module(tmp_path, "geom", "first")
    body = f"""
    import importlib.machinery as machinery, importlib.util
    path = {str(module)!r}
    loader = machinery.SourceFileLoader("geom", path)
    plain = loader.source_to_code
    def instrumented(data, path, *args, **kwargs):
        return plain(data + b"\\nINSTRUMENTED = 1\\n", path)
    loader.source_to_code = instrumented
    spec = importlib.util.spec_from_file_location("geom", path, loader=loader)
    geom = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geom)
    value = hasattr(geom, "INSTRUMENTED")
    """
    assert _record(tmp_path, body, pycache=str(tmp_path / "pycache"))["value"] is True
    assert not (tmp_path / "pycache").exists()


def test_what_an_instrumenting_loader_reads_is_recorded(tmp_path):
    module = _module(tmp_path, "geom", "first")
    config = tmp_path / "transform.toml"
    config.write_text("on = true\n")
    body = f"""
    import importlib.machinery as machinery, importlib.util
    class Configured(machinery.SourceFileLoader):
        def source_to_code(self, data, path, *args, **kwargs):
            open({str(config)!r}).read()
            return super().source_to_code(data, path, *args, **kwargs)
    path = {str(module)!r}
    spec = importlib.util.spec_from_file_location("geom", path, loader=Configured("geom", path))
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    """
    reads = _record(tmp_path, body, pycache=str(tmp_path / "pycache"))["record"]["reads"]
    assert reads[str(config)] == _sha(config)


def test_replacing_an_entry_counts_only_the_difference(tmp_path):
    cache = code_reads.CodeCache(str(tmp_path))
    data = b"VALUE = 1\n"
    digest = hashlib.sha256(data).hexdigest()
    for _ in range(3):
        cache.store(digest, "m", compile(data, "m", "exec"))
    assert cache._total == Path(cache.entry(digest, "m")).stat().st_size


def test_a_compiler_replaced_before_the_recorder_loaded_is_never_cached(tmp_path):
    """A .pth or sitecustomize wrapping the stdlib's compiler, even under its name."""
    module = _module(tmp_path, "geom", "first")
    script = f"""
    import functools, importlib.machinery as machinery, importlib.util, json, sys
    plain = machinery.SourceFileLoader.source_to_code
    @functools.wraps(plain)
    def instrumented(self, data, path, *args, **kwargs):
        return plain(self, data + b"\\nINSTRUMENTED = 1\\n", path, *args, **kwargs)
    machinery.SourceFileLoader.source_to_code = instrumented
    from strata.notebook import code_reads
    with code_reads.recording([], pycache={str(tmp_path / "pycache")!r}):
        sys.path.insert(0, {str(module.parent)!r})
        import geom
    print(json.dumps(hasattr(geom, "INSTRUMENTED")))
    """
    assert _script(tmp_path, script) is True
    assert not (tmp_path / "pycache").exists()


def test_replacing_another_processs_entry_counts_the_cache_afresh(tmp_path):
    ours, theirs = code_reads.CodeCache(str(tmp_path)), code_reads.CodeCache(str(tmp_path))

    def store(cache: code_reads.CodeCache, name: str) -> str:
        data = f"VALUE = {name!r}\n".encode()
        digest = hashlib.sha256(data).hexdigest()
        cache.store(digest, name, compile(data, name, "exec"))
        return cache.entry(digest, name)

    store(ours, "x")
    damaged = Path(store(theirs, "y"))  # written after ours counted
    damaged.write_bytes(b"damaged")
    store(ours, "y")
    assert ours._total == sum(p.stat().st_size for p in tmp_path.iterdir())
