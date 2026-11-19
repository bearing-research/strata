"""What a scratchpad cell's run read, by content: the code it loaded, the files it opened,
the directories it listed, and the places an import looked and found nothing.

Loaded by path into the harnesses, so stdlib only. Each is recorded when it happens, not
after the run: a module's hash is of the bytes the loader compiled (its ``.pyc`` is never
read; the code compiled from them is kept under that hash, :class:`CodeCache`), a file's
of what it held when first opened. At the end each is observed again, and
one that changed makes the run unstable: its result is kept but never found by its key.
Nothing under the interpreter's stdlib or environment is recorded; the environment's
identity covers those.
"""

from __future__ import annotations

import hashlib
import importlib.machinery as machinery
import importlib.util
import inspect
import json
import marshal
import os
import site
import stat
import struct
import sys
import sysconfig
import tempfile
import threading
import time
import zipimport
from collections.abc import Callable, Generator, Iterable
from contextlib import AbstractContextManager, contextmanager, nullcontext, suppress
from typing import Any

ABSENT = "absent"
# Past this a file is observed by its stat fingerprint, not read again in full.
LARGE_BYTES = 64 * 1024 * 1024
# A write this close to an observation may leave every stat field as it was (git's
# "racily clean"): such a file is hashed, never judged by its stat.
RACY_NS = 2_000_000_000
# Past this many paths the record would not be kept whole: the run is not cached.
PATH_CAP = 10_000
# Compiled code kept by its source's hash: least recently used goes past this.
PYCACHE_BYTES = 256 * 1024 * 1024
# An entry: magic (with the format's version), the header's length, the header, the
# marshalled code, then the sha256 of all of that.
_PYC_MAGIC = b"STRATAPYC\x01"
_PYC_LENGTHS = struct.Struct(">I")
_PYC_DIGEST = 32

# They change where imports resolve: a scratchpad's environment identity holds them, and a
# cell's ``# @env`` may not set them.
IMPORT_VARIABLES = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSAFEPATH",
    "PYTHONNOUSERSITE",
    "PYTHONPLATLIBDIR",
)

_SYSTEM = ("/dev", "/proc", "/sys")
# The path finder's order: extension modules, then source, then bytecode.
_SUFFIXES = (
    *machinery.EXTENSION_SUFFIXES,
    *machinery.SOURCE_SUFFIXES,
    *machinery.BYTECODE_SUFFIXES,
)
_WRITES = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
# Loaders whose loads are recorded (patched here) or hold no file of the project's.
_KNOWN_LOADERS = (
    machinery.SourceFileLoader,
    machinery.SourcelessFileLoader,
    machinery.ExtensionFileLoader,
    machinery.BuiltinImporter,
    machinery.FrozenImporter,
    machinery.NamespaceLoader,
    zipimport.zipimporter,
)


def environment_paths() -> list[str]:
    """Where the interpreter's own code lives: the stdlib, and the environment's packages
    (the whole virtual environment when it is one)."""
    paths = sysconfig.get_paths()
    found = [paths["stdlib"], paths["platstdlib"], paths["purelib"], paths["platlib"]]
    base = sysconfig.get_paths(vars={"base": sys.base_prefix, "platbase": sys.base_exec_prefix})
    found += [base["stdlib"], base["platstdlib"]]
    if sys.prefix != sys.base_prefix:
        found.append(sys.prefix)
    # What the interpreter put on its own path from its install: the stdlib's zip, its
    # lib-dynload.
    bases = [os.path.realpath(p) for p in {sys.base_prefix, sys.base_exec_prefix}]
    found += [
        entry
        for entry in sys.path
        if entry
        and any(os.path.realpath(entry).startswith(b.rstrip(os.sep) + os.sep) for b in bases)
    ]
    found += site.getsitepackages()
    user = site.getusersitepackages()
    if user:
        found.append(user)
    # Strata's own: the harness, and this file, loaded from it.
    found.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return found


def fingerprint(path: str, taken_ns: int) -> list[int] | None:
    """*path*'s stat fingerprint, ``[dev, ino, size, mtime_ns, ctime_ns, taken_ns]``, or
    None if it cannot be stat'ed."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, taken_ns]


def racy(print_: list[int]) -> bool:
    """Whether a write may have followed its observation unseen by these fields."""
    _dev, _ino, _size, mtime, ctime, taken = print_
    return ctime == 0 or max(mtime, ctime) >= taken - RACY_NS


def observe(path: str, *, listing: bool = False) -> tuple[str, list[int] | None]:
    """What is at *path* now, and its stat fingerprint (None when absent).

    A file is ``sha256:<hex>`` of its bytes, or past :data:`LARGE_BYTES` ``stat:...`` of
    its fingerprint, whenever it is observed, so one file has one form. A directory listed
    is ``list:<hex>`` of its sorted names but the dot ones, as a glob sees it.
    """
    taken = time.time_ns()
    try:
        st = os.stat(path)
    except OSError:
        return ABSENT, None
    print_ = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, taken]
    try:
        if stat.S_ISDIR(st.st_mode):
            if not listing:
                return "dir", print_
            names = sorted(n for n in os.listdir(path) if not n.startswith("."))
            digest = hashlib.sha256("\0".join(names).encode(errors="surrogateescape"))
            return f"list:{digest.hexdigest()}", print_
        if st.st_size > LARGE_BYTES:
            return "stat:" + ":".join(map(str, print_[:5])), print_
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            while chunk := file.read(1 << 20):
                digest.update(chunk)
    except OSError:
        return ABSENT, None
    return f"sha256:{digest.hexdigest()}", print_


class Reads:
    """What one run read, as recorded."""

    def __init__(self, excluded: Iterable[str]) -> None:
        self.excluded = tuple(
            sorted({os.path.realpath(p) for p in [*excluded, *environment_paths()] if p})
        )
        self.seen: dict[str, str] = {}
        self.stat: dict[str, list[int]] = {}
        self.listed: set[str] = set()
        self.written: set[str] = set()
        self.unstable: str | None = None
        self.modules_before = frozenset(sys.modules)
        self._hooks: tuple[list[Any], list[Any]] = ([], [])
        # (class, method) the recording patches.
        self.patched: list[tuple[type, str]] = []
        self._own = threading.local()
        self._tracked: dict[str, bool] = {}
        # Whether this thread is the read audit's own work (listing a native writer's
        # target), which is not the cell's either.
        self.audit_quiet: Callable[[], bool] = lambda: False
        # Makes the read audit's hook skip this thread: the code cache's files are neither.
        self.audit_hushed: Callable[[], AbstractContextManager[Any]] = nullcontext
        self.code_cache: CodeCache | None = None

    @property
    def quiet(self) -> bool:
        return getattr(self._own, "on", False) or self.audit_quiet()

    @contextmanager
    def own(self) -> Generator[None]:
        """This thread's opens are the recorder's own, not the cell's."""
        was, self._own.on = getattr(self._own, "on", False), True
        try:
            yield
        finally:
            self._own.on = was

    def tracked(self, path: str) -> bool:
        known = self._tracked.get(path)
        if known is None:
            real = os.path.realpath(path)
            known = not any(
                real == p or real.startswith(p.rstrip(os.sep) + os.sep)
                for p in (*self.excluded, *_SYSTEM)
            )
            self._tracked[path] = known
        return known

    def _unstable(self, reason: str) -> None:
        if self.unstable is None:
            self.unstable = reason

    def _put(self, path: str, seen: str, print_: list[int] | None) -> None:
        if path in self.seen or path in self.written:
            return
        if len(self.seen) >= PATH_CAP:
            self._unstable(f"the cell read more than {PATH_CAP} paths")
            return
        self.seen[path] = seen
        if print_ is not None:
            self.stat[path] = print_

    def file(self, path: str) -> None:
        if path in self.seen or path in self.written or not self.tracked(path):
            return
        with self.own():
            seen, print_ = observe(path)
        if seen == "dir":
            return  # opened to list it or to remove it: its listing, if any, is recorded
        self._put(path, seen, print_)

    def module(self, path: str, digest: str) -> None:
        """A module's file, by the sha256 of the bytes its loader runs."""
        taken = time.time_ns()
        self._put(path, f"sha256:{digest}", fingerprint(path, taken))

    def compiled(self, data: bytes, digest: str, path: str, compile_: Callable[[], Any]) -> Any:
        """The code of a module's source *data*: from the cache, else *compile_*."""
        if self.code_cache is None:
            return compile_()
        with self.own(), self.audit_hushed():
            return self.code_cache.code(digest, path, compile_)

    def absent(self, path: str) -> None:
        self._put(path, ABSENT, None)

    def listing(self, path: str) -> None:
        if path in self.listed or not self.tracked(path):
            return
        with self.own():
            seen, print_ = observe(path, listing=True)
        self.listed.add(path)
        self._put(path, seen, print_)

    def native_wrote(self, path: str, is_dir: bool) -> None:
        """A file a native writer wrote, or a folder it made: the run's own."""
        if is_dir:
            self.made(path)
        else:
            self.opened(path, "w", None)

    def made(self, path: Any) -> None:
        """A directory the cell made: its own, as a file it wrote is."""
        if not isinstance(path, int):
            self.written.add(os.path.abspath(os.fsdecode(path)))

    def opened(self, path: Any, mode: Any, flags: Any) -> None:
        if isinstance(path, int):
            return  # a descriptor: seen when it was opened
        name = os.path.abspath(os.fsdecode(path))
        writes = isinstance(mode, str) and any(c in mode for c in "wax+")
        if writes or (isinstance(flags, int) and flags & _WRITES):
            if name in self.seen and name not in self.written:
                # What it read is not what a later run will read.
                self._unstable(f"the cell wrote {name}, which it had read")
            # The run's own output, not what it read: dropped, as the run makes it again.
            self.written.add(name)
            self.seen.pop(name, None)
            self.stat.pop(name, None)
            return
        self.file(name)

    def hooks_changed(self) -> None:
        """Whether code of the cell's own (not the environment's, as six's importer is) put
        an import hook in place, whose resolution the record does not model."""
        meta, hooks = self._hooks
        now = [f for f in sys.meta_path if not isinstance(f, _Absences)]
        added = next((f for f in now if f not in meta and not self._installed(f)), None)
        if added is not None:
            self._unstable(f"the cell changed sys.meta_path ({_name(added)})")
            return
        hook = next((h for h in sys.path_hooks if h not in hooks and not self._installed(h)), None)
        if hook is not None:
            self._unstable(f"the cell changed sys.path_hooks ({_name(hook)})")

    def _installed(self, hook: Any) -> bool:
        """Whether *hook* is the environment's code, which its identity covers."""
        # A path hook is often a function or a class; a meta path finder an instance.
        owner = hook if isinstance(hook, type) or inspect.isfunction(hook) else type(hook)
        module = sys.modules.get(getattr(owner, "__module__", None) or "")
        origin = getattr(module, "__file__", None)
        return (
            module is not None
            and module.__name__ != "__main__"
            and isinstance(origin, str)
            and not self.tracked(origin)
        )

    def _recorded_by(self, loader: Any) -> bool:
        """Whether a load by *loader* was recorded: one of the patched loaders with the
        patched method, or one that loads no file."""
        kind = loader if isinstance(loader, type) else type(loader)
        for cls, attr in self.patched:
            if issubclass(kind, cls):
                # Not overridden on the way down to the patched class.
                for k in kind.__mro__:
                    if k is cls:
                        return True
                    if attr in k.__dict__:
                        return False
        return issubclass(kind, _KNOWN_LOADERS)

    def finish(self) -> dict[str, Any]:
        """Observe each path again; return the record. A failure makes it unstable, never
        the run's failure."""
        try:
            return self._finish()
        except Exception as exc:
            return {"reads": {}, "stat": {}, "unstable": f"the record failed: {exc!r}"}

    def _finish(self) -> dict[str, Any]:
        self.hooks_changed()
        for name in sorted(set(sys.modules) - self.modules_before):
            spec = getattr(sys.modules.get(name), "__spec__", None)
            loader = getattr(spec, "loader", None)
            origin = getattr(spec, "origin", None)
            if (
                loader is not None
                and not self._recorded_by(loader)
                and isinstance(origin, str)
                and os.path.isabs(origin)
                and self.tracked(origin)
            ):
                self._unstable(f"the cell loaded {name} through {_name(loader)}")
        taken = time.time_ns()
        with self.own():
            for path, seen in self.seen.items():
                if path in self.written:
                    continue
                # Read again only when its stat moved, or it was read too soon after a write.
                recorded, current = self.stat.get(path), fingerprint(path, taken)
                if seen == ABSENT and current is None:
                    continue
                if (
                    recorded is not None
                    and current is not None
                    and current[:5] == recorded[:5]
                    and not racy(recorded)
                ):
                    continue
                now, _ = observe(path, listing=path in self.listed)
                if now != seen:
                    self._unstable(f"{path} changed while the cell ran")
                    break
        reads = {p: s for p, s in self.seen.items() if p not in self.written}
        return {
            "reads": reads,
            "stat": {p: s for p, s in self.stat.items() if p in reads},
            "unstable": self.unstable,
        }


class CodeCache:
    """Code compiled from a module's source, kept under the source's sha256.

    The key holds the path too (compiled into ``co_filename``, so into tracebacks), the
    interpreter's bytecode magic and cache tag, and the optimization level. A hit is
    unmarshalled only once the entry's checksum and header match what was just derived:
    a changed byte can still unmarshal into code computing something else. Anything
    else is deleted and compiled again; deleting the whole cache is always safe.
    """

    def __init__(self, directory: str, limit: int = PYCACHE_BYTES) -> None:
        self.directory = directory
        self.limit = limit
        # Bytes this process believes the cache holds; found by one listing, then added to.
        self._total: int | None = None

    def header(self, digest: str, path: str) -> bytes:
        fields = {
            "magic": importlib.util.MAGIC_NUMBER.hex(),
            "tag": sys.implementation.cache_tag,
            "optimize": sys.flags.optimize,
            "source": digest,
            "path": path,
        }
        return json.dumps(fields, sort_keys=True).encode()

    def entry(self, digest: str, path: str) -> str:
        return os.path.join(self.directory, hashlib.sha256(self.header(digest, path)).hexdigest())

    def code(self, digest: str, path: str, compile_: Callable[[], Any]) -> Any:
        header = self.header(digest, path)
        entry = os.path.join(self.directory, hashlib.sha256(header).hexdigest())
        found = self._load(entry, header)
        if found is not None:
            return found
        code = compile_()
        self._store(entry, header, marshal.dumps(code))
        return code

    def _load(self, entry: str, header: bytes) -> Any:
        try:
            with open(entry, "rb") as f:
                blob = f.read()
        except OSError:
            # Absent (a miss) or unreadable here: compiled from source.
            return None
        payload = _pyc_payload(blob, header)
        if payload is not None:
            try:
                code = marshal.loads(payload)
            except (EOFError, ValueError, TypeError):
                code = None
            if code is not None:
                with suppress(OSError):  # only the eviction order would be off
                    os.utime(entry)  # recently used: the last to go
                return code
        # Damaged: the code compiled now replaces it.
        return None

    def _store(self, entry: str, header: bytes, payload: bytes) -> None:
        blob = _PYC_MAGIC + _PYC_LENGTHS.pack(len(header)) + header + payload
        blob += hashlib.sha256(blob).digest()
        try:
            os.makedirs(self.directory, exist_ok=True)
            fd, temp = tempfile.mkstemp(dir=self.directory, prefix=".tmp-")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(blob)
                os.replace(temp, entry)
            except OSError:
                os.unlink(temp)
                raise
        except OSError:
            # Not writable here: the next run compiles it again.
            return
        if self._total is None:
            self._total = self._size()
        else:
            self._total += len(blob)
        if self._total > self.limit:
            self._evict()

    def _listing(self) -> list[tuple[int, int, str]]:
        found = []
        try:
            with os.scandir(self.directory) as entries:
                for item in entries:
                    try:
                        st = item.stat()
                    except OSError:
                        continue  # removed meanwhile
                    found.append((st.st_mtime_ns, st.st_size, item.path))
        except OSError:
            return []
        return found

    def _size(self) -> int:
        return sum(size for _, size, _ in self._listing())

    def _evict(self) -> None:
        entries = sorted(self._listing())
        total = sum(size for _, size, _ in entries)
        for _, size, path in entries:
            if total <= self.limit:
                break
            try:
                os.unlink(path)
            except OSError:
                continue  # another process took it
            total -= size
        self._total = total


def _pyc_payload(blob: bytes, header: bytes) -> bytes | None:
    """The marshalled code in an entry, if its checksum holds and its header is *header*."""
    fixed = len(_PYC_MAGIC) + _PYC_LENGTHS.size
    if len(blob) < fixed + _PYC_DIGEST or not blob.startswith(_PYC_MAGIC):
        return None
    (header_length,) = _PYC_LENGTHS.unpack_from(blob, len(_PYC_MAGIC))
    if hashlib.sha256(blob[:-_PYC_DIGEST]).digest() != blob[-_PYC_DIGEST:]:
        return None
    if blob[fixed : fixed + header_length] != header:
        return None
    return blob[fixed + header_length : -_PYC_DIGEST]


_code_caches: dict[str, CodeCache] = {}


def code_cache(directory: str) -> CodeCache:
    """The process's cache over *directory*: a warm worker keeps one across its runs."""
    found = _code_caches.get(directory)
    if found is None:
        found = _code_caches[directory] = CodeCache(directory)
    return found


def _name(thing: Any) -> str:
    kind = thing if isinstance(thing, type) else type(thing)
    return f"{kind.__module__}.{kind.__qualname__}"


_current: Reads | None = None
# The audit hook, added once a process: one cannot be removed.
_hook_added = False


def _on_event(event: str, args: tuple[Any, ...]) -> None:
    reads = _current
    if reads is None or reads.quiet:
        return
    try:
        if event == "open":
            reads.opened(*args[:3])
        elif event == "os.mkdir":
            reads.made(args[0])
        elif event in ("os.listdir", "os.scandir"):
            path = args[0] if args[0] is not None else "."
            if not isinstance(path, int):
                reads.listing(os.path.abspath(os.fsdecode(path)))
    except Exception as exc:
        # The cell's own call must not fail on the record; the run is not cached instead.
        reads._unstable(f"the record failed: {exc!r}")


class _Absences:
    """First on ``sys.meta_path``, finding nothing itself: it records, in the order the
    path finder looks, each place an import looked under a tracked directory and found
    nothing, up to the one it found."""

    def __init__(self, reads: Reads) -> None:
        self.reads = reads

    def find_spec(self, name: str, path: Any, target: Any = None) -> None:
        reads = self.reads
        if reads.quiet:
            return None
        try:
            self._record(name, path)
        except Exception as exc:
            # The cell's import must not fail on the record; the run is not cached instead.
            reads._unstable(f"the record failed: {exc!r}")
        return None

    def _record(self, name: str, path: Any) -> None:
        reads = self.reads
        reads.hooks_changed()
        if path is None and (
            machinery.BuiltinImporter.find_spec(name) or machinery.FrozenImporter.find_spec(name)
        ):
            return  # found before any path is searched
        tail = name.rpartition(".")[2]
        with reads.own():
            searched = (
                os.fsdecode(e) if isinstance(e, bytes) else e
                for e in (sys.path if path is None else path)
            )
            entries = [
                (entry, os.path.abspath(entry or os.getcwd()))
                for entry in searched
                if isinstance(entry, str)
            ]
            # Past the last tracked entry nothing is recorded, as an environment package's
            # submodule (its own __path__) searches nothing tracked at all.
            tracked = [i for i, (_, base) in enumerate(entries) if reads.tracked(base)]
            last = tracked[-1] if tracked else -1
            for entry, base in entries[: last + 1]:
                if self._search(name, tail, entry, base):
                    return

    def _search(self, name: str, tail: str, entry: str, base: str) -> bool:
        """Whether *entry* holds the module, recording what it lacks on the way."""
        reads = self.reads
        if os.path.isfile(base):
            # An archive (a zip): observed whole, as a file read.
            reads.file(base)
            importer = sys.path_importer_cache.get(entry)
            try:
                return importer is not None and importer.find_spec(name) is not None
            except Exception:
                return False
        importer = sys.path_importer_cache.get(entry)
        if (
            importer is not None
            and not isinstance(importer, machinery.FileFinder)
            and not reads._installed(importer)
        ):
            reads._unstable(f"{entry} is served by {_name(importer)}")
        track = reads.tracked(base)
        package = os.path.join(base, tail)
        if os.path.isdir(package):
            for suffix in _SUFFIXES:
                init = os.path.join(package, "__init__" + suffix)
                if os.path.isfile(init):
                    return True
                if track:
                    reads.absent(init)
            # A namespace portion: the search goes on, for a module that would win.
        elif track:
            reads.absent(package)
        for suffix in _SUFFIXES:
            module = package + suffix
            if os.path.isfile(module):
                return True
            if track:
                reads.absent(module)
        return False


def _patch(reads: Reads) -> list[tuple[type, str, Any]]:
    """Make the file loaders record what they load; return what to restore."""
    source_get_code = machinery.SourceFileLoader.get_code
    sourceless_get_code = machinery.SourcelessFileLoader.get_code
    create_extension = machinery.ExtensionFileLoader.create_module
    # Private, and the same since 3.3: how a FileFinder lists the directory it searches.
    fill_cache = machinery.FileFinder.__dict__["_fill_cache"]

    def get_source_code(self: Any, fullname: str) -> Any:
        path = self.get_filename(fullname)
        if reads.quiet or not reads.tracked(path):
            return source_get_code(self, fullname)
        # Compiled from the bytes hashed, never from a .pyc that may predate them.
        with reads.own():
            data = self.get_data(path)
        digest = hashlib.sha256(data).hexdigest()
        reads.module(path, digest)
        return reads.compiled(data, digest, path, lambda: self.source_to_code(data, path))

    def get_sourceless_code(self: Any, fullname: str) -> Any:
        path = self.get_filename(fullname)
        if not reads.quiet and reads.tracked(path):
            with reads.own():
                reads.module(path, hashlib.sha256(self.get_data(path)).hexdigest())
        return sourceless_get_code(self, fullname)

    def create_module(self: Any, spec: Any) -> Any:
        if not reads.quiet and reads.tracked(self.path):
            reads.file(os.path.abspath(self.path))
        return create_extension(self, spec)

    def fill_finder_cache(self: Any) -> Any:
        # The import system listing a directory to search it: the searches are recorded.
        with reads.own():
            return fill_cache(self)

    patched: list[tuple[type, str, Any]] = [
        (machinery.SourceFileLoader, "get_code", get_source_code),
        (machinery.SourcelessFileLoader, "get_code", get_sourceless_code),
        (machinery.ExtensionFileLoader, "create_module", create_module),
    ]
    reads.patched = [(cls, attr) for cls, attr, _ in patched]
    patched.append((machinery.FileFinder, "_fill_cache", fill_finder_cache))
    restore: list[tuple[type, str, Any]] = [
        (cls, attr, cls.__dict__.get(attr)) for cls, attr, _ in patched
    ]
    for cls, attr, fn in patched:
        setattr(cls, attr, fn)
    return restore


@contextmanager
def for_run(manifest: dict[str, Any], observed: Any) -> Generator[Reads | None]:
    """A harness run's recording: a scratchpad cell's (its manifest has ``code_reads``),
    else None.

    The run's own directory (its inputs and folders) is left out, and the files a native
    reader reads come from the read audit's *observed*, which no open event shows.
    """
    config = manifest.get("code_reads")
    if config is None:
        yield None
        return
    anchor = config.get("anchor")
    inserted = bool(anchor) and (not sys.path or sys.path[0] != anchor)
    if inserted:
        # As under ``python -c`` in the project: its own modules import, and shadow.
        sys.path.insert(0, anchor)
    # A run's own folders (``# @output_dir``) are mounts with no URI; a loop's live
    # outside its run directory.
    own = [m["local_path"] for m in (manifest.get("mounts") or {}).values() if not m.get("uri")]
    try:
        with recording(
            [*config["exclude"], manifest.get("output_dir") or "", *own],
            pycache=config.get("pycache"),
        ) as reads:
            reads.audit_quiet = lambda: observed.quiet
            reads.audit_hushed = observed.hushed
            observed.on_native_read = reads.file
            observed.on_native_write = reads.native_wrote
            try:
                yield reads
            finally:
                observed.on_native_read = None
                observed.on_native_write = None
    finally:
        if inserted and anchor in sys.path:
            # Only the cell resolves against the project: what the harness imports after
            # (serializing its outputs) is recorded nowhere.
            sys.path.remove(anchor)


@contextmanager
def recording(excluded: Iterable[str], *, pycache: str | None = None) -> Generator[Reads]:
    """Record the body's reads into the :class:`Reads` it yields; nothing under
    *excluded* (or the interpreter's own code) is recorded. Code compiled from tracked
    source is kept in *pycache*, when given.

    The audit hook stays installed after: one cannot be removed. The finder and the
    loader patches are undone.
    """
    global _current, _hook_added
    if not _hook_added:
        sys.addaudithook(_on_event)
        _hook_added = True
    reads = Reads(excluded)
    if pycache:
        reads.code_cache = code_cache(pycache)
    finder = _Absences(reads)
    sys.meta_path.insert(0, finder)
    reads._hooks = (
        [f for f in sys.meta_path if f is not finder],
        list(sys.path_hooks),
    )
    restore = _patch(reads)
    _current = reads
    try:
        yield reads
    finally:
        _current = None
        for cls, attr, original in restore:
            if original is None:
                delattr(cls, attr)
            else:
                setattr(cls, attr, original)
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)


def warm_list(notebook_dir: str) -> list[str]:
    """What a warm worker imports before its cell: the notebook's declared dependencies."""
    import tomllib

    try:
        with open(os.path.join(notebook_dir, "pyproject.toml"), "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return []
    result = []
    for dep in data.get("project", {}).get("dependencies", []):
        name = dep.split("[")[0].split(";")[0]
        for sep in (">=", "==", "!=", "~=", "<", ">"):
            name = name.split(sep)[0]
        result.append(name.strip())
    return result


def scratch_warm_list(anchor: str, notebook_dir: str) -> list[str]:
    """A scratchpad's warm list (design 4.2): what its project and the scratchpad itself
    declare, as the modules those distributions install."""
    import importlib.metadata as metadata
    import re

    def canonical(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    provides: dict[str, list[str]] = {}
    for module, dists in metadata.packages_distributions().items():
        for dist in dists:
            provides.setdefault(canonical(dist), []).append(module)
    modules: list[str] = []
    for dep in [*warm_list(anchor), *warm_list(notebook_dir)]:
        found = provides.get(canonical(dep)) or [dep.replace("-", "_")]
        modules += [m for m in sorted(found) if m.isidentifier() and not m.startswith("_")]
    return list(dict.fromkeys(modules))


def _top_level(names: Iterable[str]) -> set[str]:
    """The top-level ones of *names* found on the path, which the anchor could shadow."""
    return {
        name.partition(".")[0]
        for name in names
        if not name.startswith("_nb_")
        and getattr(getattr(sys.modules.get(name), "__spec__", None), "has_location", False)
    }


def _warm_up(warm: Iterable[str], excluded: list[str], anchor: str) -> tuple[list[str], list[str]]:
    """Import each of *warm* under the recorder; return those whose import loaded no
    tracked code (none outside *excluded*, the installed environment) and nothing the
    *anchor* shadows, and those that did."""
    kept: list[str] = []
    dropped: list[str] = []
    before = set(sys.modules)
    # Loaded by a dropped import: an entry found already loaded was not checked itself.
    tainted: set[str] = set()
    for name in warm:
        if name in before:
            # The harness's own: loaded before any cell either way.
            kept.append(name)
            continue
        if name in sys.modules:
            # Loaded by an earlier entry's import, so only as clean as that one.
            (dropped if name.partition(".")[0] in tainted else kept).append(name)
            continue
        loaded_before = set(sys.modules)
        with recording(excluded) as reads:
            try:
                __import__(name)
            except Exception:  # noqa: BLE001 - a worker that fails to import it has not loaded it
                loaded = False
            else:
                loaded = True
        record = reads.finish()
        new = _top_level(set(sys.modules) - loaded_before)
        # Under ``python -c`` in the project, a file there would be what these import.
        shadowed = any(state != ABSENT for state in base_candidates(anchor, new).values())
        tracked = any(state != ABSENT for state in record["reads"].values())
        if record["unstable"] or tracked or shadowed:
            # Neither would show in any run's record: the cell imports it, and records it.
            dropped.append(name)
            tainted |= new
        elif loaded:
            kept.append(name)
    return kept, dropped


def watch_state(path: str, mode: bool | None) -> str:
    """What :func:`probe_environment` watches at *path*: its listing (``True``), its bytes
    (``False``), or (``None``) its stat alone, for a file too big to read on every key."""
    if mode is not None:
        return observe(path, listing=mode)[0]
    try:
        st = os.stat(path)
    except OSError:
        return ABSENT
    return f"stat:{st.st_dev}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}"


def _installs(path: str) -> bool:
    try:
        return any(n.endswith(".dist-info") for n in os.listdir(path))
    except OSError:
        return False


def _direct_url(dist: Any) -> dict[str, Any] | None:
    import json

    try:
        found = json.loads(dist.read_text("direct_url.json") or "null")
    except ValueError:
        return None
    return found if isinstance(found, dict) else None


def base_candidates(anchor: str, names: Iterable[str]) -> dict[str, str]:
    """Where a top-level import of each of *names* would look first, under *anchor* at
    ``sys.path[0]``, and what is there now: one that is no longer absent shadows a module
    loaded before the cell started, which no import of its records."""
    found: dict[str, str] = {}
    for name in names:
        package = os.path.join(anchor, name)
        paths = [package]
        paths += [os.path.join(package, "__init__" + suffix) for suffix in _SUFFIXES]
        paths += [package + suffix for suffix in _SUFFIXES]
        for path in paths:
            found[path] = observe(path)[0]
    return found


def base_names_to_check(anchor: str, names: Iterable[str], base: dict[str, str]) -> list[str]:
    """Those of *names* with something at one of their places under *anchor* when *base*
    was recorded: a change in one (an edit, an ``__init__.py`` added to a folder there)
    leaves the anchor's own listing as it was."""
    present = {path for path, state in base.items() if state != ABSENT}
    found = []
    for name in names:
        package = os.path.join(anchor, name)
        if package in present or any(package + s in present for s in _SUFFIXES):
            found.append(name)
    return found


def probe_environment(
    anchor: str,
    sites: list[str],
    notebook_dir: str | None = None,
    warm: list[str] | None = None,
) -> dict[str, Any]:
    """A scratchpad's environment identity E's parts, as a cell started here would find
    them (design 4.2): the distributions D, the interpreter build, the import resolution I,
    the base record B; and what to observe to know they still hold.

    *sites* are the project's site-packages an overlay adds. The base image is imported
    here first, as before a scratchpad's cell: *warm*, else :func:`scratch_warm_list` of
    *notebook_dir*, less what loads tracked code (``warm_dropped``; the caller probes
    again with ``warm``). Run as the harness, so ``sys.path`` and the modules loaded
    before a cell are a cell's.
    """
    import hashlib as _hashlib
    import importlib.metadata as metadata
    import platform
    import re

    path = [anchor, *(p for p in sys.path if p != anchor)]
    # A folder installed into (a dist-info in it) a ``.pth`` put on the path; never one on
    # PYTHONPATH, whose code is the user's: a run records what it reads there.
    chosen = set(filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep)))
    roots = sorted(
        set(site.getsitepackages())
        | set(sites)
        | {p for p in sys.path if p and p not in chosen and _installs(p)}
    )
    if warm is None:
        warm = scratch_warm_list(anchor, notebook_dir) if notebook_dir else []
    kept, dropped = _warm_up(warm, [*roots, *([notebook_dir] if notebook_dir else [])], anchor)
    watched: dict[str, bool | None] = {}
    for root in roots:
        watched[root] = True
        try:
            pth = sorted(n for n in os.listdir(root) if n.endswith(".pth"))
        except OSError:
            pth = []
        watched.update((os.path.join(root, n), False) for n in pth)
    # What can rewrite the path and the hooks at startup, wherever site would find it.
    for entry in path[1:]:
        for name in ("sitecustomize.py", "usercustomize.py"):
            watched.setdefault(os.path.join(entry, name), False)
    watched[os.path.join(sys.prefix, "pyvenv.cfg")] = False
    if notebook_dir:
        # Their dependencies are the warm list.
        watched[os.path.join(notebook_dir, "pyproject.toml")] = False
        watched[os.path.join(anchor, "pyproject.toml")] = False
    # The build: an interpreter upgraded in place is a new binary and standard library.
    watched[os.path.realpath(sys.executable)] = None
    watched[os.path.realpath(os.__file__)] = None

    distributions: set[str] = set()
    editables: list[dict[str, Any]] = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        url = _direct_url(dist)
        if url and url.get("dir_info", {}).get("editable"):
            # Its code is keyed by what runs read; where it points, and how, is here.
            files = {}
            for entry in dist.files or ():
                base = os.path.basename(str(entry))
                located = str(dist.locate_file(entry))
                if base.endswith(".pth") or base.startswith("__editable__"):
                    files[base] = observe(located)[0]
                    watched[located] = False
                elif base == "direct_url.json":
                    watched[located] = False
            editables.append({"name": canonical, "url": url.get("url"), "files": files})
            continue
        entry = canonical + "==" + dist.version
        if url:
            # From a URL, a VCS or a path: one version can be any code, so what it installed.
            record = (dist.read_text("RECORD") or "").encode()
            entry += " @" + _hashlib.sha256(record).hexdigest()
            # A reinstall at the same version leaves every listing as it was.
            for listed in dist.files or ():
                if listed.name == "RECORD" and listed.parent.name.endswith(".dist-info"):
                    watched[str(dist.locate_file(listed))] = False
        distributions.add(entry)

    user = site.getusersitepackages()
    watched[user] = True
    # Loaded before the cell: a top-level one found on the path, which the anchor could
    # shadow (a builtin or frozen one it cannot).
    names = sorted(
        {
            name.partition(".")[0]
            for name, module in list(sys.modules.items())
            if not name.startswith("_nb_")
            and getattr(getattr(module, "__spec__", None), "has_location", False)
        }
    )
    identity = {
        "distributions": sorted(distributions),
        "build": [
            sys.implementation.name,
            platform.python_version(),
            getattr(sys, "abiflags", ""),
            sysconfig.get_platform(),
        ],
        "path": path,
        "meta_path": [_name(finder) for finder in sys.meta_path],
        "path_hooks": [_name(hook) for hook in sys.path_hooks],
        "editables": sorted(editables, key=lambda e: e["name"]),
        "variables": {k: os.environ.get(k) for k in IMPORT_VARIABLES},
        "user_site": [bool(site.ENABLE_USER_SITE), user in sys.path],
        "base": base_candidates(anchor, names),
        "warm": kept,
    }
    return {
        "identity": identity,
        "observed": {p: [mode, watch_state(p, mode)] for p, mode in watched.items()},
        "anchor_listing": observe(anchor, listing=True)[0],
        "base_names": names,
        "roots": roots,
        "warm": kept,
        "warm_dropped": dropped,
    }
