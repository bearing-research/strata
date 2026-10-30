"""The channel a running cell publishes checkpoints through (ADR 002 E4b).

The cell calls ``strata.checkpoint(**parts)``: the parts are written under the run
directory and one JSON line tells the executor, which holds the build and the ledger, to
release them. A checkpoint of files (``strata.checkpoint_files``, what
``vestigo.torch.HostedRun`` saves through) is whatever a writer leaves in the
sequence's folder. A cold harness writes the line on a pipe the executor passed
(``STRATA_EVENTS_FD``, the lifeline's pattern in the other direction); a warm worker
writes it on its protocol stream ahead of its result line. The cell never waits for
acceptance. Stdlib only: loaded by path into the notebook venv, like the client.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

EVENTS_FD_ENV = "STRATA_EVENTS_FD"
EVENT_KEY = "event"
CHECKPOINTS_DIR = "checkpoints"
"""Under the run directory: one folder per checkpoint sequence."""
FILES_PART = "__files__"
"""The part name a checkpoint of files is stored under; no ``checkpoint()`` part may take it."""

_UNSET = object()


def is_event_line(line: str) -> bool:
    """Whether a protocol line is a run event rather than a worker's result."""
    return line.startswith('{"' + EVENT_KEY + '"')


class RunChannel:
    """What ``strata.checkpoint`` / ``restore`` / ``head`` do inside the cell's process."""

    def __init__(
        self,
        write: Callable[[str], None],
        run_dir: Path,
        *,
        serialize: Callable[[Any, Path, str], dict[str, Any]],
        deserialize: Callable[[str, Path], Any],
        restore: dict[str, Any] | None = None,
        close: Callable[[], None] | None = None,
    ) -> None:
        self._write = write
        self._run_dir = Path(run_dir)
        self._serialize = serialize
        self._deserialize = deserialize
        self._restore = restore
        self._close = close
        # A resumed attempt continues the build's sequence.
        self._sequence = int(restore.get("sequence", 0)) if restore else 0
        self._restored: Any = _UNSET

    def checkpoint(self, **parts: Any) -> int:
        if not parts:
            raise ValueError("checkpoint() needs at least one named part")
        if FILES_PART in parts:
            raise ValueError(f"{FILES_PART} is not a part name a checkpoint can take")
        sequence = self._sequence + 1
        folder = self._run_dir / CHECKPOINTS_DIR / str(sequence)
        folder.mkdir(parents=True, exist_ok=True)
        specs: dict[str, dict[str, Any]] = {}
        for name, value in parts.items():
            spec = dict(self._serialize(value, folder, name))
            specs[name] = {
                "content_type": spec["content_type"],
                "file": f"{CHECKPOINTS_DIR}/{sequence}/{spec['file']}",
                "bytes": spec.get("bytes"),
            }
        self._emit({EVENT_KEY: "checkpoint", "sequence": sequence, "parts": specs})
        self._sequence = sequence
        return sequence

    def checkpoint_files(self, step: int, write: Callable[[Path], object]) -> int:
        if not isinstance(step, int) or isinstance(step, bool):
            raise ValueError(f"step must be an int, not {step!r}")
        sequence = self._sequence + 1
        folder = self._run_dir / CHECKPOINTS_DIR / str(sequence)
        # A write that failed under this sequence left files no release names.
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        write(folder)
        files = _regular_files(folder)
        if not files:
            raise ValueError("checkpoint_files() wrote no files")
        self._emit({EVENT_KEY: "checkpoint", "sequence": sequence, "step": step, "files": files})
        self._sequence = sequence
        return sequence

    def restore_files(self) -> Path | None:
        if self._restore is None:
            return None
        if "folder" not in self._restore:
            raise RuntimeError(
                f"release {self._restore['release_id']} holds no checkpoint files; "
                "strata.restore() returns its parts"
            )
        return self._run_dir / self._restore["folder"]

    def restore(self) -> dict[str, Any] | None:
        if self._restore is None:
            return None
        if self._restored is _UNSET:
            self._restored = {
                name: self._deserialize(spec["content_type"], self._run_dir / spec["file"])
                for name, spec in self._restore["parts"].items()
            }
        return self._restored

    def head(self, name: str, sequence: int) -> None:
        if not name or "/" in name:
            raise ValueError(f"head name must be a plain name, not {name!r}")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
            raise ValueError(f"sequence must be a checkpoint sequence, not {sequence!r}")
        self._emit({EVENT_KEY: "head", "name": name, "sequence": sequence})

    def close(self) -> None:
        if self._close is not None:
            self._close()
            self._close = None

    def _emit(self, payload: dict[str, Any]) -> None:
        self._write(json.dumps(payload, sort_keys=False) + "\n")


def _regular_files(root: Path) -> list[str]:
    """Each regular file under *root* as a POSIX path relative to it; links are skipped."""
    found: list[str] = []
    for dirpath, _, names in os.walk(root):
        for name in names:
            path = Path(dirpath, name)
            if stat.S_ISREG(path.lstat().st_mode):
                found.append(path.relative_to(root).as_posix())
    return sorted(found)


def claim_pipe() -> int | None:
    """Keep the event pipe named in the environment from the cell's children.

    Called first thing in the harness, whether or not a client is injected: a child
    that inherited the write end would hold the executor's read open after the
    harness exits.
    """
    raw = os.environ.get(EVENTS_FD_ENV)
    if not raw:
        return None
    fd = int(raw)
    os.set_inheritable(fd, False)
    return fd


def from_pipe(
    manifest: dict[str, Any],
    *,
    serialize: Callable[[Any, Path, str], dict[str, Any]],
    deserialize: Callable[[str, Path], Any],
) -> RunChannel | None:
    """A cold harness's channel over the pipe named in the environment, or None without one."""
    fd = claim_pipe()
    if fd is None:
        return None
    stream = os.fdopen(fd, "w", encoding="utf-8")

    def write(line: str) -> None:
        stream.write(line)
        stream.flush()

    return RunChannel(
        write,
        Path(manifest["output_dir"]),
        serialize=serialize,
        deserialize=deserialize,
        restore=manifest.get("restore"),
        close=stream.close,
    )


def over_stream(
    manifest: dict[str, Any],
    stream: Any,
    *,
    serialize: Callable[[Any, Path, str], dict[str, Any]],
    deserialize: Callable[[str, Path], Any],
) -> RunChannel:
    """A warm worker's channel: lines on its protocol *stream* ahead of the result line."""

    def write(line: str) -> None:
        stream.write(line)
        stream.flush()

    return RunChannel(
        write,
        Path(manifest["output_dir"]),
        serialize=serialize,
        deserialize=deserialize,
        restore=manifest.get("restore"),
    )
