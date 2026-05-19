"""Environment backend protocol — abstracts per-notebook env management.

Phase 1 (this commit): ``EnvironmentBackend`` is the seam every env
mutation flows through. ``UvBackend`` is the only implementation; it
wraps the pre-existing ``_run_uv_command`` / ``run_uv_command_streaming``
helpers in ``dependencies.py`` so test mocks of those helpers continue
to work unchanged. This is structural — zero behavior change.

Phase 2 (separate diff): an ``AttachedBackend`` will be added for users
whose notebook lives in a pre-existing pip-managed venv. Detection at
session open, ``[environment] backend = "..."`` override in
``notebook.toml``, and UI signaling for the read-only mutation
surface all land in that phase.

Why a backend protocol at all: the notebook executes cells in the
venv at ``<notebook_dir>/.venv/bin/python`` and mutates that venv via
``uv add``/``uv remove``/``uv sync``. The execution side is
backend-agnostic (a venv interpreter is a venv interpreter); the
mutation side is uv-specific. The protocol separates "I attach to
this venv" from "I manage this venv" so the next backend doesn't
have to fake mutating operations it can't honestly support.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from strata.notebook.dependencies import _UvCommandResult

_StreamCallback = Callable[[str, str, bool], Awaitable[None] | None]


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
    the attached-venv mode planned for Phase 2; those raise a clear
    error if mutation methods are called anyway."""

    def python_executable(self) -> Path:
        """Absolute path to the interpreter cell-execution should use.

        Backends that own the venv compute this from
        ``<notebook_dir>/.venv/bin/python``; attached backends return
        whichever interpreter the user pointed Strata at.
        """
        ...

    def sync(self, *, python_version: str | None, timeout: int) -> _UvCommandResult:
        """Reconcile the venv against the declared dependencies.

        ``python_version`` (when provided) re-pins the venv to that
        interpreter as part of the sync.
        """
        ...

    def add(self, package: str, *, timeout: int) -> _UvCommandResult:
        """Add a package to the declared dependencies and sync."""
        ...

    def remove(self, package: str, *, timeout: int) -> _UvCommandResult:
        """Remove a package from the declared dependencies and sync."""
        ...

    async def sync_streaming(
        self,
        *,
        python_version: str | None,
        timeout: int,
        on_update: _StreamCallback | None,
    ) -> _UvCommandResult:
        """Streaming variant of ``sync`` for the background job loop."""
        ...

    async def add_streaming(
        self,
        package: str,
        *,
        timeout: int,
        on_update: _StreamCallback | None,
    ) -> _UvCommandResult:
        """Streaming variant of ``add``."""
        ...

    async def remove_streaming(
        self,
        package: str,
        *,
        timeout: int,
        on_update: _StreamCallback | None,
    ) -> _UvCommandResult:
        """Streaming variant of ``remove``."""
        ...

    async def lock_streaming(
        self,
        *,
        timeout: int,
        on_update: _StreamCallback | None,
    ) -> _UvCommandResult:
        """Regenerate the lockfile from declared dependencies (no venv
        materialization). Used during ``requirements.txt`` /
        ``environment.yaml`` imports where the caller has already
        written ``pyproject.toml`` and just needs the lockfile to
        catch up."""
        ...


class UvBackend:
    """Uv-driven environment backend — the only Strata-managed mode.

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


def get_backend(notebook_dir: Path) -> EnvironmentBackend:
    """Resolve the env backend for *notebook_dir*.

    Phase 1: always returns a ``UvBackend``. Phase 2 will branch on
    detection (presence of ``uv.lock`` vs. attached venv) and on the
    ``[environment] backend = "..."`` override in ``notebook.toml``.
    Keeping the resolution behind one helper means callers don't
    accumulate construction logic at every site that needs a backend.
    """
    return UvBackend(notebook_dir)
