"""Environment backend protocol — abstracts per-notebook env management.

Two implementations:

* ``UvBackend`` — Strata manages the venv via ``uv add``/``remove``/``sync``.
  Wraps the pre-existing ``_run_uv_command`` / ``run_uv_command_streaming``
  helpers in ``dependencies.py`` so test mocks keep working.
* ``AttachedBackend`` — Strata attaches to an existing pip-managed venv
  read-only. Mutation methods raise ``BackendDoesNotSupportMutations``;
  the route layer turns those into ``409 Conflict`` and the UI signals
  read-only-ness via the env status payload.

Backend resolution lives at ``get_backend(notebook_dir)``. Order of
precedence:

1. ``notebook.toml [strata] backend = "uv" | "attached"`` override.
2. Detection from on-disk evidence (see ``detect_backend``).

The override lives under ``[strata]`` rather than ``[environment]``
because the parser strips ``[environment]`` aggressively (the block
historically held legacy runtime metadata that has since migrated
to ``.strata/runtime.json``). ``[strata]`` is a fresh namespace for
Strata-specific notebook-level configuration.

The protocol separates "I attach to this venv" from "I manage this
venv" so the next backend doesn't have to fake mutating operations it
can't honestly support.
"""

from __future__ import annotations

import logging
import tomllib
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from strata.notebook.dependencies import _UvCommandResult

logger = logging.getLogger(__name__)

_StreamCallback = Callable[[str, str, bool], Awaitable[None] | None]


class BackendDoesNotSupportMutations(RuntimeError):
    """Raised when a mutating op (``add``/``remove``/``sync``/...) is
    called on a backend whose ``supports_mutations`` is ``False``.

    The route layer catches this and returns ``409 Conflict``; the
    frontend disables mutation buttons up front so users don't reach
    the error in normal flows.
    """


class EnvironmentBackend(Protocol):
    """Per-notebook environment management surface.

    Implementations own the venv at ``<notebook_dir>/.venv/`` and
    handle the four user-driven mutations (``add``, ``remove``,
    ``sync``, ``set_python_version``). Both sync and streaming
    variants exist because the foreground REST endpoints want the
    final result while the background env-job system streams progress
    to the UI.
    """

    name: str
    """Human-readable backend label, e.g. ``"uv"``. Surfaced in job
    snapshots and environment status so the UI can show "Powered by
    uv" / "Attached venv" indicators."""

    supports_mutations: bool
    """Whether ``add``/``remove``/``sync``/``set_python_version`` are
    callable on this backend. ``False`` for read-only backends like
    ``AttachedBackend``; those raise
    ``BackendDoesNotSupportMutations`` if mutation methods are called
    anyway."""

    def python_executable(self) -> Path:
        """Absolute path to the interpreter cell-execution should use.

        Backends that own the venv compute this from
        ``<notebook_dir>/.venv/bin/python``; attached backends return
        whichever interpreter the user pointed Strata at.
        """
        ...

    def sync(self, *, python_version: str | None, timeout: int) -> _UvCommandResult: ...
    def add(self, package: str, *, timeout: int) -> _UvCommandResult: ...
    def remove(self, package: str, *, timeout: int) -> _UvCommandResult: ...

    async def sync_streaming(
        self,
        *,
        python_version: str | None,
        timeout: int,
        on_update: _StreamCallback | None,
    ) -> _UvCommandResult: ...

    async def add_streaming(
        self,
        package: str,
        *,
        timeout: int,
        on_update: _StreamCallback | None,
    ) -> _UvCommandResult: ...

    async def remove_streaming(
        self,
        package: str,
        *,
        timeout: int,
        on_update: _StreamCallback | None,
    ) -> _UvCommandResult: ...

    async def lock_streaming(
        self,
        *,
        timeout: int,
        on_update: _StreamCallback | None,
    ) -> _UvCommandResult: ...


class UvBackend:
    """Uv-driven environment backend — the Strata-managed mode.

    Wraps ``dependencies._run_uv_command`` and
    ``dependencies.run_uv_command_streaming`` so test mocks that patch
    those module-level helpers continue to function. The bounded
    output buffers, error shaping, and timeout handling all stay where
    they were; this class is the seam, not the implementation.
    """

    name = "uv"
    supports_mutations = True

    def __init__(self, notebook_dir: Path) -> None:
        self.notebook_dir = Path(notebook_dir)

    def python_executable(self) -> Path:
        """Return the venv interpreter path.

        Returns the conventional ``.venv/bin/python`` location even if
        the venv hasn't been materialized yet — callers that need a
        "does it actually exist" check should ``.exists()`` the
        returned path themselves.
        """
        return self.notebook_dir / ".venv" / "bin" / "python"

    def sync(self, *, python_version: str | None, timeout: int) -> _UvCommandResult:
        from strata.notebook.dependencies import _run_uv_command

        args = ["sync"]
        if python_version:
            args += ["--python", python_version]
        return _run_uv_command(self.notebook_dir, args, timeout=timeout, display_name="uv sync")

    def add(self, package: str, *, timeout: int) -> _UvCommandResult:
        from strata.notebook.dependencies import _run_uv_command

        return _run_uv_command(
            self.notebook_dir, ["add", package], timeout=timeout, display_name="uv add"
        )

    def remove(self, package: str, *, timeout: int) -> _UvCommandResult:
        from strata.notebook.dependencies import _run_uv_command

        return _run_uv_command(
            self.notebook_dir,
            ["remove", package],
            timeout=timeout,
            display_name="uv remove",
        )

    async def sync_streaming(
        self,
        *,
        python_version: str | None,
        timeout: int,
        on_update: _StreamCallback | None,
    ) -> _UvCommandResult:
        from strata.notebook.dependencies import run_uv_command_streaming

        args = ["sync"]
        if python_version:
            args += ["--python", python_version]
        return await run_uv_command_streaming(
            self.notebook_dir,
            args,
            timeout=timeout,
            display_name="uv sync",
            on_update=on_update,
        )

    async def add_streaming(
        self,
        package: str,
        *,
        timeout: int,
        on_update: _StreamCallback | None,
    ) -> _UvCommandResult:
        from strata.notebook.dependencies import run_uv_command_streaming

        return await run_uv_command_streaming(
            self.notebook_dir,
            ["add", package],
            timeout=timeout,
            display_name="uv add",
            on_update=on_update,
        )

    async def remove_streaming(
        self,
        package: str,
        *,
        timeout: int,
        on_update: _StreamCallback | None,
    ) -> _UvCommandResult:
        from strata.notebook.dependencies import run_uv_command_streaming

        return await run_uv_command_streaming(
            self.notebook_dir,
            ["remove", package],
            timeout=timeout,
            display_name="uv remove",
            on_update=on_update,
        )

    async def lock_streaming(
        self,
        *,
        timeout: int,
        on_update: _StreamCallback | None,
    ) -> _UvCommandResult:
        from strata.notebook.dependencies import run_uv_command_streaming

        return await run_uv_command_streaming(
            self.notebook_dir,
            ["lock"],
            timeout=timeout,
            display_name="uv lock",
            on_update=on_update,
        )


class AttachedBackend:
    """Read-only attached-venv backend — for users whose notebook lives
    inside an existing pip/poetry/virtualenv project.

    Strata uses the venv's interpreter for cell execution and reads
    the resolved-dependency state for the UI, but does not mutate.
    Every mutating call raises ``BackendDoesNotSupportMutations``; the
    route layer maps that to ``409 Conflict`` and the UI keeps the
    add/remove/sync buttons disabled with a tooltip pointing the user
    at their own tooling.
    """

    name = "attached"
    supports_mutations = False

    def __init__(self, notebook_dir: Path) -> None:
        self.notebook_dir = Path(notebook_dir)

    def python_executable(self) -> Path:
        """Return the existing venv's interpreter.

        Detection only routes here when ``.venv/bin/python`` actually
        exists, so this path is real on disk. Cell execution uses it
        the same way it would for uv -- the interpreter doesn't care
        which tool created the venv.
        """
        return self.notebook_dir / ".venv" / "bin" / "python"

    def _refuse(self, op: str) -> _UvCommandResult:
        raise BackendDoesNotSupportMutations(
            f"AttachedBackend cannot {op}: this notebook is attached to a venv "
            "Strata does not manage. Use your own tooling (pip / poetry / etc.) "
            "to mutate the environment, or set "
            '``[strata]\nbackend = "uv"`` in notebook.toml to let Strata '
            "take it over."
        )

    def sync(self, *, python_version: str | None, timeout: int) -> _UvCommandResult:
        del python_version, timeout
        return self._refuse("sync")

    def add(self, package: str, *, timeout: int) -> _UvCommandResult:
        del package, timeout
        return self._refuse("add")

    def remove(self, package: str, *, timeout: int) -> _UvCommandResult:
        del package, timeout
        return self._refuse("remove")

    async def sync_streaming(
        self,
        *,
        python_version: str | None,
        timeout: int,
        on_update: _StreamCallback | None,
    ) -> _UvCommandResult:
        del python_version, timeout, on_update
        return self._refuse("sync")

    async def add_streaming(
        self,
        package: str,
        *,
        timeout: int,
        on_update: _StreamCallback | None,
    ) -> _UvCommandResult:
        del package, timeout, on_update
        return self._refuse("add")

    async def remove_streaming(
        self,
        package: str,
        *,
        timeout: int,
        on_update: _StreamCallback | None,
    ) -> _UvCommandResult:
        del package, timeout, on_update
        return self._refuse("remove")

    async def lock_streaming(
        self,
        *,
        timeout: int,
        on_update: _StreamCallback | None,
    ) -> _UvCommandResult:
        del timeout, on_update
        return self._refuse("lock")


# Backend literal used in the ``notebook.toml [environment] backend``
# override and in API payloads so the frontend can match on it.
BackendName = Literal["uv", "attached"]


def _read_backend_override(notebook_dir: Path) -> BackendName | None:
    """Return the ``notebook.toml [strata] backend`` override or None.

    The override always wins over detection so a user can force
    Strata to manage an existing venv (set to "uv") or hand off
    control of a uv-created venv to their own tooling (set to
    "attached"). Invalid values are ignored with a warning rather
    than crashing the open path -- a typo in notebook.toml should
    not prevent the user from opening their work.
    """
    notebook_toml = notebook_dir / "notebook.toml"
    if not notebook_toml.exists():
        return None
    try:
        with open(notebook_toml, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    strata_section = data.get("strata")
    if not isinstance(strata_section, dict):
        return None
    backend = strata_section.get("backend")
    if backend in ("uv", "attached"):
        return backend
    if backend is not None:
        logger.warning(
            "Unknown [strata] backend %r in %s; falling back to detection",
            backend,
            notebook_toml,
        )
    return None


def _venv_has_uv_marker(notebook_dir: Path) -> bool:
    """Return whether ``.venv/pyvenv.cfg`` was written by uv.

    uv stamps a ``uv = X.Y.Z`` line into pyvenv.cfg from at least
    0.4 onward; stdlib ``venv`` and ``virtualenv`` don't. That single
    line is the definitive venv-layer signal — see the detection
    algorithm comment for the full layered check.
    """
    pyvenv = notebook_dir / ".venv" / "pyvenv.cfg"
    if not pyvenv.exists():
        return False
    try:
        contents = pyvenv.read_text(encoding="utf-8")
    except OSError:
        return False
    for raw in contents.splitlines():
        stripped = raw.strip()
        if not stripped or "=" not in stripped:
            continue
        key, _, _ = stripped.partition("=")
        if key.strip() == "uv":
            return True
    return False


def detect_backend(notebook_dir: Path) -> BackendName:
    """Return the inferred backend from on-disk evidence.

    Layered check:

    1. ``uv.lock`` at the project root → ``"uv"`` (uv has been run on
       this project; the lockfile is the project-level signal that
       survives venv deletion).
    2. ``.venv/pyvenv.cfg`` exists and contains the ``uv =`` marker
       → ``"uv"`` (uv created this venv; lockfile may be missing but
       intent is clear).
    3. ``.venv/pyvenv.cfg`` exists *without* the marker
       → ``"attached"`` (some other tool created the venv).
    4. No venv at all → ``"uv"`` (new notebook; Strata will create
       a uv-managed venv on first sync).

    The override at ``[environment] backend`` in ``notebook.toml``
    always wins over detection.
    """
    if (notebook_dir / "uv.lock").exists():
        return "uv"
    pyvenv = notebook_dir / ".venv" / "pyvenv.cfg"
    if pyvenv.exists():
        return "uv" if _venv_has_uv_marker(notebook_dir) else "attached"
    return "uv"


def get_backend(notebook_dir: Path) -> EnvironmentBackend:
    """Resolve the env backend for *notebook_dir*.

    Order of precedence: ``notebook.toml`` override → on-disk
    detection. Keeping the resolution behind one helper means callers
    don't accumulate construction logic at every site that needs a
    backend.
    """
    override = _read_backend_override(notebook_dir)
    chosen = override if override is not None else detect_backend(notebook_dir)
    if chosen == "attached":
        return AttachedBackend(notebook_dir)
    return UvBackend(notebook_dir)
