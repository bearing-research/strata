"""What a scratchpad cell's run read, by content: the code it loaded, the files it opened,
the directories it listed, and the places an import looked and found nothing.

Loaded by path into the harnesses, so stdlib only. Each is recorded when it happens, not
after the run: a module's hash is of the bytes the loader compiled (its ``.pyc`` is never
read), a file's of what it held when first opened. At the end each is observed again, and
one that changed makes the run unstable: its result is kept but never found by its key.
Nothing under the interpreter's stdlib or environment is recorded; the environment's
identity covers those.
"""

from __future__ import annotations

import hashlib
import importlib.machinery as machinery
import inspect
import os
import site
import stat
import sys
import sysconfig
import threading
import time
import zipimport
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from typing import Any

ABSENT = "absent"
# Past this a file is observed by its stat fingerprint, not read again in full.
LARGE_BYTES = 64 * 1024 * 1024
# A write this close to an observation may leave every stat field as it was (git's
# "racily clean"): such a file is hashed, never judged by its stat.
RACY_NS = 2_000_000_000
# Past this many paths the record would not be kept whole: the run is not cached.
PATH_CAP = 10_000

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

    def module(self, path: str, data: bytes) -> None:
        """A module's file, as the bytes its loader runs."""
        taken = time.time_ns()
        self._put(path, f"sha256:{hashlib.sha256(data).hexdigest()}", fingerprint(path, taken))

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
        reads.module(path, data)
        return self.source_to_code(data, path)

    def get_sourceless_code(self: Any, fullname: str) -> Any:
        path = self.get_filename(fullname)
        if not reads.quiet and reads.tracked(path):
            with reads.own():
                reads.module(path, self.get_data(path))
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
    # A run's own folders (``# @output_dir``) are mounts with no URI; a loop's live
    # outside its run directory.
    own = [m["local_path"] for m in (manifest.get("mounts") or {}).values() if not m.get("uri")]
    with recording([*config["exclude"], manifest.get("output_dir") or "", *own]) as reads:
        reads.audit_quiet = lambda: observed.quiet
        observed.on_native_read = reads.file
        observed.on_native_write = reads.native_wrote
        try:
            yield reads
        finally:
            observed.on_native_read = None
            observed.on_native_write = None


@contextmanager
def recording(excluded: Iterable[str]) -> Generator[Reads]:
    """Record the body's reads into the :class:`Reads` it yields; nothing under
    *excluded* (or the interpreter's own code) is recorded.

    The audit hook stays installed after: one cannot be removed. The finder and the
    loader patches are undone.
    """
    global _current, _hook_added
    if not _hook_added:
        sys.addaudithook(_on_event)
        _hook_added = True
    reads = Reads(excluded)
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
