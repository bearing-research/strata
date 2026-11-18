"""What a scratchpad cell's run read, as ``code_reads`` records it (design 4.3).

Each case runs its body in a fresh interpreter: the recorder patches the import system and
installs an audit hook that cannot be removed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from strata.notebook import code_reads
from strata.notebook.code_reads import ABSENT

_RUN = """\
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("_code_reads", {module!r})
code_reads = importlib.util.module_from_spec(spec)
spec.loader.exec_module(code_reads)
with code_reads.recording({exclude!r}) as reads:
{body}
record = reads.finish()
print(json.dumps({{"record": record, "value": globals().get("value")}}))
"""


def _record(tmp_path: Path, body: str, *, exclude: list[str] | None = None) -> dict:
    script = _RUN.format(
        module=code_reads.__file__,
        exclude=exclude or [],
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
    # Only where the import looked before the stdlib: the start directory, first on the
    # path under `python -c`, where a decimal.py would shadow it.
    assert reads
    assert all(Path(path).parent == tmp_path and seen == ABSENT for path, seen in reads.items())
    assert str(tmp_path / "decimal.py") in reads


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
import importlib.util, json, types
spec = importlib.util.spec_from_file_location("_code_reads", {code_reads.__file__!r})
code_reads = importlib.util.module_from_spec(spec)
spec.loader.exec_module(code_reads)
observed = types.SimpleNamespace(on_native_read=None)
manifest = {{"code_reads": {{"exclude": []}}, "output_dir": {str(run)!r}}}
with code_reads.for_run(manifest, observed) as reads:
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
