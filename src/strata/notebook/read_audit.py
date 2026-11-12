"""What a cell's run read that nothing it declares covers.

Loaded by path into the harnesses, so stdlib only. A cell's key holds what it declares
(its source, upstream values, the lock, mounts, ``@table``, ``@fetch``, ``@dataset``); a
file it reads around them is invisible to the key, and its cached result outlives the
file's change. This is an audit, never a key: an ``sys.addaudithook`` hook records the
files a cell opens and the directories it lists, a ``sys.modules`` diff the modules it
loads, and the programs it starts, and :func:`report` sorts out what no declaration
covers. Native readers and writers (pyarrow, polars, duckdb) open files without an
``open`` event, so their entry points that take a path are wrapped as their modules are
imported (``_NATIVE_READERS``, ``_NATIVE_WRITERS``) and record it.
"""

from __future__ import annotations

import ast
import functools
import glob
import hashlib
import inspect
import os
import stat
import sys
import threading
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NotRequired, TypedDict

# Past this many paths a run's record stops growing and says it is partial.
PATH_CAP = 10_000
# Per finding, the paths a result carries.
ITEM_CAP = 200

UNDECLARED_READ = "undeclared_read"
UNDECLARED_LISTING = "undeclared_listing"
UNDECLARED_IMPORT = "undeclared_import"
UNLOCKED_PACKAGE = "unlocked_package"
WORK_DIR_READ = "work_dir_read"
SUBPROCESS_RUN = "subprocess_run"
# The audit itself raised: the run stands, and says what it could not check.
AUDIT_FAILED = "audit_failed"

# The codes whose items are paths, which a local run fingerprints.
_PATH_CODES = frozenset({UNDECLARED_READ, UNDECLARED_LISTING, UNDECLARED_IMPORT, WORK_DIR_READ})

_WRITES = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_APPEND

# Entry points that open a path in native code, where the hook sees no ``open``: (module,
# attribute, the parameter holding the path, the one holding the mode). A class is
# wrapped at its ``__init__``; one missing in the installed version is skipped. pandas
# reads Parquet through pyarrow. DuckDB SQL strings and the immutable types
# (``pyarrow.OSFile``, ``pyarrow.csv.CSVWriter``) are out of reach.
_NATIVE_READERS: tuple[tuple[str, str, str, str | None], ...] = (
    ("pyarrow", "memory_map", "path", "mode"),
    ("pyarrow", "input_stream", "source", None),
    *(
        ("pyarrow.parquet", name, param, None)
        for name, param in (
            ("read_table", "source"),
            ("read_pandas", "source"),
            ("ParquetFile", "source"),
            ("ParquetDataset", "path_or_paths"),
            ("read_metadata", "where"),
            ("read_schema", "where"),
        )
    ),
    ("pyarrow.dataset", "dataset", "source", None),
    ("pyarrow.dataset", "parquet_dataset", "metadata_path", None),
    ("pyarrow.feather", "read_table", "source", None),
    ("pyarrow.feather", "read_feather", "source", None),
    ("pyarrow.orc", "read_table", "source", None),
    ("pyarrow.orc", "ORCFile", "source", None),
    ("pyarrow.csv", "read_csv", "input_file", None),
    ("pyarrow.csv", "open_csv", "input_file", None),
    ("pyarrow.json", "read_json", "input_file", None),
    ("pyarrow.json", "open_json", "input_file", None),
    ("pyarrow.ipc", "open_file", "source", None),
    ("pyarrow.ipc", "open_stream", "source", None),
    *(
        ("polars", name, "source", None)
        for name in (
            "read_parquet",
            "scan_parquet",
            "read_parquet_schema",
            "read_parquet_metadata",
            "read_csv",
            "scan_csv",
            "read_ndjson",
            "scan_ndjson",
            "read_json",
            "read_ipc",
            "read_ipc_stream",
            "read_ipc_schema",
            "scan_ipc",
            "read_lines",
            "scan_lines",
            "read_avro",
            "read_excel",
            "read_ods",
            "read_delta",
            "scan_delta",
        )
    ),
    *(
        ("duckdb", f"{owner}{name}", "path_or_buffer", None)
        for owner in ("", "DuckDBPyConnection.")
        for name in ("read_parquet", "from_parquet", "read_csv", "from_csv_auto", "read_json")
    ),
)
# (module, attribute, the parameter holding the path): what they write is the run's own.
_NATIVE_WRITERS: tuple[tuple[str, str, str], ...] = (
    ("pyarrow", "output_stream", "source"),
    ("pyarrow", "create_memory_map", "path"),
    ("pyarrow.parquet", "write_table", "where"),
    ("pyarrow.parquet", "ParquetWriter", "where"),
    ("pyarrow.parquet", "write_to_dataset", "root_path"),
    ("pyarrow.parquet", "write_metadata", "where"),
    ("pyarrow.dataset", "write_dataset", "base_dir"),
    ("pyarrow.feather", "write_feather", "dest"),
    ("pyarrow.orc", "write_table", "where"),
    ("pyarrow.orc", "ORCWriter", "where"),
    ("pyarrow.csv", "write_csv", "output_file"),
    ("pyarrow.ipc", "new_file", "sink"),
    ("pyarrow.ipc", "new_stream", "sink"),
    # It opens pyarrow's own stream, which the hook does not see.
    ("pandas", "DataFrame.to_parquet", "path"),
    *(
        ("polars", f"DataFrame.write_{name}", "file")
        for name in ("parquet", "csv", "ipc", "ipc_stream", "ndjson", "json", "avro")
    ),
    ("polars", "DataFrame.write_excel", "workbook"),
    ("polars", "DataFrame.write_delta", "target"),
    *(("polars", f"LazyFrame.sink_{name}", "path") for name in ("parquet", "csv", "ipc", "ndjson")),
    ("polars", "LazyFrame.sink_delta", "target"),
    *(
        ("duckdb", f"DuckDBPyRelation.{name}", "file_name")
        for name in ("write_parquet", "to_parquet", "write_csv", "to_csv")
    ),
)
_NATIVE_MODULES = frozenset(module for module, *_ in _NATIVE_READERS + _NATIVE_WRITERS)

# What a dataset's discovery skips in a folder (pyarrow's ``ignore_prefixes``): ``_SUCCESS``,
# ``_delta_log``, ``.crc`` files are a writer's bookkeeping, not data read.
_BOOKKEEPING = ("_", ".")


class ReadFinding(TypedDict):
    """One kind of input a run read that its cell does not declare."""

    code: str
    # Sorted: paths, a package's dist names, or program names. At most ITEM_CAP.
    items: list[str]
    # How many there were; more than ``items`` holds past the cap.
    total: int
    # The record stopped at PATH_CAP, so there may be more than ``total``.
    partial: bool
    # A local run's paths as it read them (:func:`fingerprint`), so a later change to one
    # can be told: only the server's own files can be checked again by the server.
    # CHANGED_DURING_RUN for one that changed while the run had it.
    fingerprints: NotRequired[dict[str, list[int]]]


# Matches no path's fingerprint: the file changed between the run's read and its end.
CHANGED_DURING_RUN: list[int] = []


class Observed:
    """What one run did, as the hook saw it."""

    def __init__(self) -> None:
        # Absolute path -> whether this run's first open of it wrote.
        self.files: dict[str, bool] = {}
        # Each path's fingerprint at its first read or listing, and the files written.
        self.read_as: dict[str, list[int] | None] = {}
        self.written: set[str] = set()
        self.listings: set[str] = set()
        self.programs: set[str] = set()
        self.modules_before = frozenset(sys.modules)
        self.work_dir = os.getcwd()
        self.partial = False
        self.quiet = False

    def _path(self, path: Any) -> str | None:
        if path is None:
            path = "."
        if isinstance(path, int):
            return None  # a file descriptor: its open was seen when it was made
        name = os.fsdecode(path)
        # As the cell meant it then: it may chdir later.
        return os.path.normpath(name) if os.path.isabs(name) else os.path.abspath(name)

    def open(self, path: Any, mode: Any, flags: Any) -> None:
        name = self._path(path)
        if name is None:
            return
        if isinstance(mode, str):
            wrote = any(c in mode for c in "wax")
            changes = wrote or "+" in mode
        else:
            wrote = isinstance(flags, int) and bool(flags & _WRITES)
            changes = wrote or (isinstance(flags, int) and bool(flags & os.O_RDWR))
        if changes:
            self.written.add(name)
        if name in self.files:
            return
        if len(self.files) >= PATH_CAP:
            self.partial = True
            return
        self.files[name] = wrote
        if not wrote:
            # The event fires ahead of the open: this is the file the run reads.
            self.read_as[name] = self._fingerprint(name)

    def native(self, source: Any, mode: str) -> None:
        """A path a native reader was given: a file, a folder of them, a glob, or a list.

        A buffer, bytes (polars reads them as the file's content) or an open file is
        skipped (its ``open`` was seen), and so is a path that is not here (a remote URI).
        A folder or a glob is listed here, which the hook records as listings.
        """
        for found in _native_paths(source, globs=True):
            for file in _files_in(found, data_only=True):
                if len(self.files) >= PATH_CAP:
                    self.partial = True
                    return
                self.open(file, mode, None)

    def files_at(self, target: Any) -> dict[str, list[int] | None]:
        """What a native writer's *target* holds now, unseen by the hook."""
        self.quiet = True
        try:
            held: dict[str, list[int] | None] = {}
            for found in _native_paths(target, globs=False):
                for file in _files_in(found, data_only=False):
                    if len(held) >= PATH_CAP:
                        self.partial = True
                        return held
                    held[file] = fingerprint(file)
            return held
        finally:
            self.quiet = False

    def native_write(self, target: Any, before: Mapping[str, list[int] | None]) -> None:
        """What a native writer made or changed at *target*: the run's own, as an ``open``
        to write is."""
        for file, now in self.files_at(target).items():
            if file not in before or before[file] != now:
                self.open(file, "w", None)

    def listing(self, path: Any) -> None:
        name = self._path(path)
        if name is None or name in self.listings:
            return
        if len(self.listings) >= PATH_CAP:
            self.partial = True
            return
        self.listings.add(name)
        self.read_as.setdefault(name, self._fingerprint(name))

    def _fingerprint(self, name: str) -> list[int] | None:
        # A directory's fingerprint lists it, which comes back through the hook.
        self.quiet = True
        try:
            return fingerprint(name)
        finally:
            self.quiet = False

    def program(self, program: Any) -> None:
        if isinstance(program, (list, tuple)):
            program = program[0] if program else None
        if isinstance(program, (str, bytes, os.PathLike)):
            text = os.fsdecode(program)
            self.programs.add(os.path.basename(text.split()[0]) if text.strip() else text)


def _native_paths(source: Any, *, globs: bool) -> list[str]:
    """The paths here a native call names: its own if it exists, else, with *globs*, what
    the pattern in it matches (polars globs). A literal ``[`` or ``?`` in a name is a
    name, as pyarrow reads it."""
    if isinstance(source, (list, tuple)):
        return [path for item in source for path in _native_paths(item, globs=globs)]
    if not isinstance(source, (str, os.PathLike)):
        return []
    text = os.fsdecode(source).removeprefix("file://")
    name = os.path.abspath(os.path.expanduser(text))
    if os.path.exists(name):
        return [name]
    if not globs or not glob.has_magic(text):
        return []
    # Only the cell's own text is a pattern, not the folder it is under.
    if text.startswith("~"):
        head, _, pattern = text.partition(os.sep)
        root = os.path.expanduser(head)
    elif os.path.isabs(text):
        root, pattern = None, text
    else:
        root, pattern = os.getcwd(), text
    found = glob.glob(pattern, root_dir=root, recursive=True)
    return sorted(os.path.join(root, path) if root else path for path in found)


def _files_in(path: str, *, data_only: bool) -> list[str]:
    """*path* if a file, else the files under it; with *data_only*, those a dataset's
    discovery reads (no ``_BOOKKEEPING`` names)."""
    if not os.path.isdir(path):
        return [path]
    found = []
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if not (data_only and d.startswith(_BOOKKEEPING)))
        found.extend(
            os.path.join(root, file)
            for file in sorted(files)
            if not (data_only and file.startswith(_BOOKKEEPING))
        )
    return found


_current: Observed | None = None
_installed = False
# Set while a wrapped native call runs: pandas calls pyarrow, which calls itself, on the
# one path, and only the outermost call records it.
_in_native = threading.local()


def _on_event(event: str, args: tuple[Any, ...]) -> None:
    observed = _current
    if observed is None or observed.quiet:
        return
    try:
        if event == "open":
            observed.open(*args[:3])
        elif event in ("os.scandir", "os.listdir"):
            observed.listing(args[0])
        elif event == "subprocess.Popen":
            observed.program(args[0] if args[0] is not None else args[1])
        elif event == "os.system":
            observed.program(args[0])
        elif event in ("os.posix_spawn", "os.exec", "os.spawn"):
            observed.program(args[1] if event == "os.spawn" else args[0])
    except Exception:
        # The cell's own call must not fail on the audit; the record is partial.
        observed.partial = True


@contextmanager
def recording() -> Generator[Observed]:
    """Record what the body does into the :class:`Observed` it yields.

    The hook stays installed after: one cannot be removed. Each harness process runs one
    cell (a batch, one at a time), so it only ever records the run in progress.
    """
    global _current, _installed
    if not _installed:
        sys.addaudithook(_on_event)
        sys.meta_path.insert(0, _WrapOnImport())
        for module in _NATIVE_MODULES & set(sys.modules):
            _wrap_natives(module)
        _installed = True
    observed = Observed()
    _current = observed
    try:
        yield observed
    finally:
        _current = None


def _path_argument(
    signature: inspect.Signature | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    param: str,
    mode: str | None,
) -> tuple[Any, str] | None:
    """The path a call names in *param*, and its mode; None for one on another filesystem
    (pyarrow's or fsspec's), which is not a local file."""
    arguments = dict(kwargs)
    if signature is not None:
        try:
            bound = signature.bind_partial(*args, **kwargs).arguments
        except TypeError:
            bound = {}  # the call itself raises
        for name, value in bound.items():
            kind = signature.parameters[name].kind
            if kind is inspect.Parameter.VAR_KEYWORD:
                arguments.update(value)
            elif kind is not inspect.Parameter.VAR_POSITIONAL:
                arguments[name] = value
    if param not in arguments and (signature is None or param not in signature.parameters):
        # A ``*args`` signature (duckdb's ``read_csv``): the path comes first, after a self.
        positional = args[1:] if signature and "self" in signature.parameters else args
        arguments[param] = positional[0] if positional else None
    filesystem = arguments.get("filesystem")
    if filesystem is not None and type(filesystem).__name__ != "LocalFileSystem":
        return None
    how = arguments.get(mode, "r") if mode else "r"
    return arguments.get(param), how if isinstance(how, str) else "r"


def _native(
    fn: Callable[..., Any], param: str, mode: str | None, *, writes: bool, at: tuple[str, str]
) -> Callable[..., Any]:
    """*fn*, recording the path its *param* names: before the call for a reader, what it
    made or changed after it for a writer. *at* is the module and name it is set at."""
    try:
        signature: inspect.Signature | None = inspect.signature(fn)
    except (TypeError, ValueError):
        signature = None

    @functools.wraps(fn)
    def native(*args: Any, **kwargs: Any) -> Any:
        observed = _current
        if observed is None or observed.quiet or getattr(_in_native, "call", False):
            return fn(*args, **kwargs)
        _in_native.call = True
        try:
            named = before = None
            try:
                named = _path_argument(signature, args, kwargs, param, mode)
                if named is not None and writes:
                    before = observed.files_at(named[0])
                elif named is not None:
                    observed.native(*named)
            except Exception:
                # The cell's own call must not fail on the audit; the record is partial.
                observed.partial = True
            result = fn(*args, **kwargs)
            if named is not None and before is not None:
                try:
                    observed.native_write(named[0], before)
                except Exception:
                    observed.partial = True
            return result
        finally:
            _in_native.call = False

    # Where pickle looks it up, so it pickles by reference, as the original did.
    native.__module__, native.__qualname__ = at
    return native


def _wrap_natives(module_name: str) -> None:
    """Wrap *module_name*'s native readers and writers in place."""
    module = sys.modules.get(module_name)
    entries = [(*reader, False) for reader in _NATIVE_READERS]
    entries += [(name, attribute, param, None, True) for name, attribute, param in _NATIVE_WRITERS]
    for name, attribute, param, mode, writes in entries:
        if name != module_name or module is None:
            continue
        *path, last = attribute.split(".")
        owner: Any = module
        for part in path:
            owner = getattr(owner, part, None)
        target = getattr(owner, last, None)
        qualname = attribute
        if isinstance(target, type):
            owner, last, target, qualname = (
                target,
                "__init__",
                target.__init__,
                f"{attribute}.__init__",
            )
        if target is None:
            continue
        wrapper = _native(target, param, mode, writes=writes, at=(module_name, qualname))
        try:
            setattr(owner, last, wrapper)
        except (AttributeError, TypeError):
            continue  # an immutable type


class _WrapOnImport:
    """A ``sys.meta_path`` finder that finds nothing itself: it wraps a native reader's or
    writer's module's entry points once the finder after it has loaded the module."""

    def find_spec(self, name: str, path: Any, target: Any = None) -> Any:
        if name not in _NATIVE_MODULES:
            return None
        for finder in sys.meta_path:
            find = getattr(finder, "find_spec", None)
            if finder is self or find is None:
                continue
            spec = find(name, path, target)
            if spec is not None:
                break
        else:
            return None
        exec_module = getattr(spec.loader, "exec_module", None)
        if exec_module is None:
            return spec

        def exec_and_wrap(module: Any) -> None:
            exec_module(module)
            _wrap_natives(name)

        try:
            spec.loader.exec_module = exec_and_wrap
        except AttributeError:
            pass  # a loader that takes no attributes: its readers stay unwrapped
        return spec


def _home(*parts: str) -> str:
    return os.path.join(os.path.expanduser("~"), *parts)


def _system_roots() -> list[str]:
    """Where an interpreter and its libraries read their own state: not a cell's input."""
    roots = [
        sys.prefix,
        sys.base_prefix,
        sys.exec_prefix,
        sys.base_exec_prefix,
        "/dev",
        "/proc",
        "/sys",
        "/etc",
        "/private/etc",
        "/usr/share",
        "/usr/lib",
        "/System",
        "/Library",
        _home("Library", "Caches"),
        _home("Library", "Preferences"),
    ]
    roots.extend(
        os.environ[key] for key in ("XDG_CACHE_HOME", "XDG_CONFIG_HOME") if key in os.environ
    )
    return roots


class _Roots:
    """Path prefixes, matched as written and with links resolved. With *hidden_under*,
    also everything in that folder's dot-folders: a home's ``.matplotlib``, ``.cache``,
    ``.aws`` hold tools' own state."""

    def __init__(self, roots: list[str], *, hidden_under: str | None = None) -> None:
        normal = {os.path.abspath(root) for root in roots if root}
        self._roots = normal | {os.path.realpath(root) for root in normal}
        self._hidden = (
            {os.path.abspath(hidden_under), os.path.realpath(hidden_under)}
            if hidden_under
            else set()
        )

    def holds(self, path: str) -> bool:
        if self._holds(path):
            return True
        real = os.path.realpath(path)
        return real != path and self._holds(real)

    def _holds(self, path: str) -> bool:
        if any(
            path == root or path.startswith(root.rstrip(os.sep) + os.sep) for root in self._roots
        ):
            return True
        for folder in self._hidden:
            rest = path[len(folder) + 1 :] if path.startswith(folder + os.sep) else ""
            if rest.startswith(".") and os.sep in rest:
                return True
        return False


def _static_imports(source: str) -> set[str]:
    """The modules *source* names in an ``import``: a batch's cell imports one an earlier
    cell already loaded, and the ``sys.modules`` diff alone would not show it."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            dotted = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            dotted = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
        else:
            continue
        for name in dotted:
            parts = name.split(".")
            names.update(".".join(parts[: i + 1]) for i in range(len(parts)))
    return names


def _module_files(observed: Observed, source: str) -> dict[str, str]:
    """Each module the run loaded or names, by its file."""
    files: dict[str, str] = {}
    names = (set(sys.modules) - observed.modules_before) | _static_imports(source)
    for name in names:
        module = sys.modules.get(name)
        file = getattr(module, "__file__", None)
        if isinstance(file, str):
            files[os.path.abspath(file)] = name
    return files


def _import_dirs() -> set[str]:
    """The directories an import lists while it looks for a module: each ``sys.path``
    entry and package, themselves, not what is under them."""
    dirs = [entry for entry in sys.path if entry]
    for module in list(sys.modules.values()):
        dirs.extend(
            path for path in getattr(module, "__path__", None) or () if isinstance(path, str)
        )
    normal = {os.path.abspath(path) for path in dirs}
    return normal | {os.path.realpath(path) for path in normal}


class _Unlocked:
    """Which installed dist, of those the lock does not hold, a module's file is from."""

    def __init__(self, scope: Mapping[str, Any]) -> None:
        self._tops: Mapping[str, str] = scope.get("unlocked") or {}
        self._site_packages = [
            (_Roots([path]), os.path.realpath(path)) for path in scope.get("site_packages") or ()
        ]
        self._roots = [
            (_Roots([root]), dist) for root, dist in (scope.get("unlocked_roots") or {}).items()
        ]

    def __bool__(self) -> bool:
        return bool(self._tops or self._roots)

    def dist(self, file: str) -> str | None:
        for roots, real in self._site_packages:
            if roots.holds(file):
                parts = os.path.relpath(os.path.realpath(file), real).split(os.sep)
                # A namespace package's dists are told apart one level down.
                return self._tops.get(parts[0]) or self._tops.get("/".join(parts[:2]))
        for roots, dist in self._roots:
            if roots.holds(file):
                return dist
        return None


def fingerprint(path: str) -> list[int] | None:
    """A file's ``[size, mtime_ns]``, a directory's ``[hash of its entry names]``, or
    None if it cannot be read.

    Neither device nor inode: a container restart or a copied notebook changes them and
    not the content. A directory's own mtime moves whenever anything in it is replaced
    (an editor's or Strata's atomic save), so a listing is its names.
    """
    try:
        st = os.stat(path)
        if not stat.S_ISDIR(st.st_mode):
            return [st.st_size, st.st_mtime_ns]
        names = "\0".join(sorted(os.listdir(path)))
    except OSError:
        return None
    return [int.from_bytes(hashlib.sha256(names.encode(errors="surrogateescape")).digest()[:8])]


def _fingerprints(observed: Observed, paths: list[str]) -> dict[str, list[int]]:
    """Each path as the run read it, or CHANGED_DURING_RUN if it changed while the run
    had it. A file the run wrote too is as the run left it; a gone one has none."""
    found: dict[str, list[int]] = {}
    for path in paths:
        now = fingerprint(path)
        if now is None:
            continue
        read_as = observed.read_as.get(path)
        if path in observed.written or read_as is None or read_as == now:
            found[path] = now
        else:
            found[path] = CHANGED_DURING_RUN
    return found


def _finding(
    code: str, items: set[str], partial: bool, *, observed: Observed | None
) -> ReadFinding:
    ordered = sorted(items)
    finding = ReadFinding(code=code, items=ordered[:ITEM_CAP], total=len(ordered), partial=partial)
    if observed is not None and code in _PATH_CODES:
        finding["fingerprints"] = _fingerprints(observed, finding["items"])
    return finding


def report(
    observed: Observed,
    *,
    source: str,
    mounts: Mapping[str, Mapping[str, Any]],
    run_dir: str,
    scope: Mapping[str, Any] | None,
) -> list[ReadFinding]:
    """What *observed* read that the cell does not declare, one finding per kind.

    Covered: its *mounts*' local paths, its *run_dir* (inputs and outputs), the
    interpreter and system state, and *scope*, the server's word on the notebook's
    environment (``audit_scope``): the import roots of its local packages, and which
    installed dists the lock does not hold. A file the run opened to write before
    reading it is its own. Without a *scope* (a remote worker's run), every installed
    dist counts as locked.
    """
    try:
        return _report(observed, source=source, mounts=mounts, run_dir=run_dir, scope=scope)
    except Exception as exc:
        return [
            ReadFinding(
                code=AUDIT_FAILED, items=[f"{type(exc).__name__}: {exc}"], total=1, partial=True
            )
        ]


def _report(
    observed: Observed,
    *,
    source: str,
    mounts: Mapping[str, Mapping[str, Any]],
    run_dir: str,
    scope: Mapping[str, Any] | None,
) -> list[ReadFinding]:
    local = scope is not None
    scope = scope or {}
    covered = _Roots(
        [str(spec.get("local_path") or "") for spec in mounts.values()]
        + [run_dir, str(Path(__file__).parent)]
        + list(scope.get("package_roots") or ())
        + _system_roots(),
        hidden_under=_home(),
    )
    work = _Roots([observed.work_dir])
    modules = _module_files(observed, source)
    module_files = set(modules)
    for name in modules.values():
        cached = getattr(getattr(sys.modules.get(name), "__spec__", None), "cached", None)
        if isinstance(cached, str):
            module_files.add(os.path.abspath(cached))

    found: dict[str, set[str]] = {}
    for path, wrote in observed.files.items():
        if wrote or path in module_files or covered.holds(path):
            continue
        found.setdefault(WORK_DIR_READ if work.holds(path) else UNDECLARED_READ, set()).add(path)

    uncovered = {p for p in observed.listings if not covered.holds(p) and not work.holds(p)}
    if uncovered:
        imports = _import_dirs()
        listed = {p for p in uncovered if p not in imports and os.path.realpath(p) not in imports}
        if listed:
            found[UNDECLARED_LISTING] = listed

    unlocked = _Unlocked(scope)
    for file in modules:
        dist = unlocked.dist(file) if unlocked else None
        if dist is not None:
            found.setdefault(UNLOCKED_PACKAGE, set()).add(dist)
        elif not covered.holds(file):
            found.setdefault(UNDECLARED_IMPORT, set()).add(file)

    if observed.programs:
        found[SUBPROCESS_RUN] = observed.programs

    paths = (UNDECLARED_READ, UNDECLARED_LISTING, WORK_DIR_READ)
    return [
        _finding(
            code, items, observed.partial and code in paths, observed=observed if local else None
        )
        for code, items in sorted(found.items())
    ]


def merge(runs: list[list[ReadFinding]]) -> list[ReadFinding]:
    """Several runs' findings as one cell's: a loop's iterations."""
    items: dict[str, set[str]] = {}
    totals: dict[str, int] = {}
    partial: dict[str, bool] = {}
    fingerprints: dict[str, dict[str, list[int]]] = {}
    for findings in runs:
        for finding in findings:
            code = finding["code"]
            items.setdefault(code, set()).update(finding["items"])
            # Past the cap a run's items are a sample, so its total is a floor.
            totals[code] = max(totals.get(code, 0), finding["total"])
            partial[code] = partial.get(code, False) or finding["partial"]
            if "fingerprints" in finding:
                # The newest run's, which a later iteration's own write leaves as it ends;
                # a change during any run stays one.
                seen = fingerprints.setdefault(code, {})
                for path, st in finding["fingerprints"].items():
                    if seen.get(path) != CHANGED_DURING_RUN:
                        seen[path] = st
    merged = []
    for code, found in sorted(items.items()):
        finding = ReadFinding(
            code=code,
            items=sorted(found)[:ITEM_CAP],
            total=max(totals[code], len(found)),
            partial=partial[code],
        )
        if code in fingerprints:
            finding["fingerprints"] = {
                path: fingerprints[code][path]
                for path in finding["items"]
                if path in fingerprints[code]
            }
        merged.append(finding)
    return merged
