"""Each cell's last run: a token minted as its code starts and again as it ends
(``.strata/runs.json``).

A cell ``@after`` another is keyed on that one's token (``keys.after_run``): only running
the upstream's code changes what it left in the world, a run that is stopped or dies
partway included, and a refusal or a cache hit never starts it. The second token, however
the run ends, makes stale what was keyed while it ran; one more follows what lands after
the code ends: a run's ``@write_table`` writes and the read-write mounts it syncs back.
The file is apart from ``runtime.json`` and written under a lock, so a run in another
process (the CLI, an agent) is read, and no other writer drops it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import filelock

from strata.notebook import nonce
from strata.notebook.quiesce import refuses_while_held

logger = logging.getLogger(__name__)


def runs_path(notebook_dir: Path) -> Path:
    return Path(notebook_dir) / ".strata" / "runs.json"


def token(notebook_dir: Path, cell_id: str) -> str:
    """*cell_id*'s last-run token, or, when it never ran here, this checkout's ``never``.

    So a cell after an upstream that never ran here is keyed on this checkout, not on a
    constant another checkout (a teammate's, the team store) shares. Read only: a key is
    worked out on the event loop and while the notebook is held.
    """
    found = read(notebook_dir).get(cell_id)
    return found if found is not None else f"never:{checkout_id(notebook_dir)}"


def checkout_id(notebook_dir: Path) -> str:
    """This checkout of the notebook: its directory on this machine. A recorder replaces it."""
    st = Path(notebook_dir).stat()
    where = f"{socket.gethostname()}:{st.st_dev}:{st.st_ino}"
    return hashlib.sha256(where.encode("utf-8")).hexdigest()


def read(notebook_dir: Path) -> dict[str, str]:
    """Each cell's last-run token, as the file holds it now."""
    try:
        return json.loads(runs_path(notebook_dir).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


# Held for one small rewrite; a holder past this is a process stopped while holding it.
LOCK_TIMEOUT_SECONDS = 30.0


@asynccontextmanager
async def running(notebook_dir: Path, cell_id: str) -> AsyncGenerator[None]:
    """Around the code of one run of *cell_id*: a token as it starts and as it ends."""
    await mint(notebook_dir, cell_id)
    try:
        yield
    finally:
        # A stopped run is the one that most needs it.
        with anyio.CancelScope(shield=True):
            await mint_after(notebook_dir, cell_id)


async def mint_after(notebook_dir: Path, cell_id: str) -> None:
    """:func:`mint` as a run ends: a failure is logged, not raised over the run's outcome.

    The run's start was recorded, so a cell after it is keyed on that token, stale from
    the moment the run began.
    """
    await anyio.to_thread.run_sync(ended, notebook_dir, cell_id)


def ended(notebook_dir: Path, cell_id: str) -> None:
    """:func:`mint_after` in this thread, for an effect that lands after the code ends."""
    try:
        # Even while the notebook is held: a run that drained past its window ends.
        _record(notebook_dir, cell_id)
    except filelock.Timeout as exc:
        logger.warning("Cell %s's run ended unrecorded: %s", cell_id, exc)
    except OSError as exc:
        logger.warning("Cell %s's run ended unrecorded: %s", cell_id, exc)


async def mint(notebook_dir: Path, cell_id: str) -> str:
    """:func:`record`, off the event loop: the lock may wait on another process."""
    return await anyio.to_thread.run_sync(record, notebook_dir, cell_id)


@refuses_while_held
def record(notebook_dir: Path, cell_id: str) -> str:
    """Mint *cell_id*'s token for a run whose code starts now; the token."""
    return _record(notebook_dir, cell_id)


def _record(notebook_dir: Path, cell_id: str) -> str:
    token = nonce.run_nonce()
    _update(notebook_dir, {cell_id: token})
    return token


@refuses_while_held
def update(notebook_dir: Path, tokens: dict[str, str]) -> None:
    """Set these cells' tokens, keeping the others'."""
    _update(notebook_dir, tokens)


def _update(notebook_dir: Path, tokens: dict[str, str]) -> None:
    path = runs_path(notebook_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = filelock.FileLock(path.with_name(path.name + ".lock"), timeout=LOCK_TIMEOUT_SECONDS)
    with lock:
        runs = {**read(notebook_dir), **tokens}
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(runs, indent=2, sort_keys=True))
            f.flush()
            # Replaced by a file that may be empty after a crash otherwise.
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
