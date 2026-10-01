"""Environment backend protocol for per-notebook env management.

Every env mutation flows through an ``EnvironmentBackend``. Cell execution only
needs a venv interpreter; mutation (``uv add``/``remove``/``sync``) is
backend-specific, which is what the protocol separates.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from strata.notebook.dependencies import _UvCommandResult

_StreamCallback = Callable[[str, str, bool], Awaitable[None] | None]


class EnvironmentBackend(Protocol):
    """Per-notebook environment management surface.

    Handles the user-driven mutations (``add``, ``remove``, ``sync``,
    ``set_python_version``). Sync and streaming variants exist because REST
    endpoints want the final result while background env jobs stream progress.
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
        """Absolute path to the interpreter cell execution should use."""
        ...

    def sync(self, *, python_version: str | None, timeout: int) -> _UvCommandResult:
        """Reconcile the venv against the declared dependencies.

        ``python_version``, when given, re-pins the venv to that interpreter.
        """
        ...

    def add(self, package: str, *, timeout: int, dev: bool = False) -> _UvCommandResult:
        """Add a package to the declared dependencies (the ``dev`` group when *dev*) and sync."""
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
        """Regenerate the lockfile from declared dependencies without syncing the venv."""
        ...


class UvBackend:
    """Uv-driven backend: each notebook gets its own ``.venv``.

    Delegates to ``dependencies._run_uv_command`` and
    ``dependencies.run_uv_command_streaming``, so tests that patch those helpers
    still apply.
    """

    name = "uv"
    supports_mutations = True

    def __init__(self, notebook_dir: Path) -> None:
        self.notebook_dir = Path(notebook_dir)

    def python_executable(self) -> Path:
        """Return ``.venv/bin/python``, whether or not the venv exists yet."""
        return self.notebook_dir / ".venv" / "bin" / "python"

    def sync(self, *, python_version: str | None, timeout: int) -> _UvCommandResult:
        from strata.notebook.dependencies import _run_uv_command

        args = ["sync"]
        if python_version:
            args += ["--python", python_version]
        return _run_uv_command(self.notebook_dir, args, timeout=timeout, display_name="uv sync")

    def add(self, package: str, *, timeout: int, dev: bool = False) -> _UvCommandResult:
        from strata.notebook.dependencies import _run_uv_command

        return _run_uv_command(
            self.notebook_dir,
            ["add", "--dev", package] if dev else ["add", package],
            timeout=timeout,
            display_name="uv add",
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

    The server-wide ``notebook_env_backend`` chooses: ``uv`` gives each notebook
    its own ``.venv``; ``shared`` links it into one environment per lockfile
    (``strata.notebook.shared_env``).
    """
    config = _config()
    if getattr(config, "notebook_env_backend", "uv") == "shared":
        from strata.notebook.shared_env import SharedEnvBackend

        return SharedEnvBackend(notebook_dir, shared_env_root(config))
    return UvBackend(notebook_dir)


def shared_root() -> Path | None:
    """The shared store's root when ``notebook_env_backend`` is ``shared``."""
    config = _config()
    if getattr(config, "notebook_env_backend", "uv") != "shared":
        return None
    return shared_env_root(config)


def shared_env_root(config: Any) -> Path:
    """Where shared environments live.

    ``notebook_shared_env_dir``, or ``envs`` beside the notebook storage directory.
    """
    configured = getattr(config, "notebook_shared_env_dir", None)
    if configured is not None:
        return Path(configured)
    return Path(config.notebook_storage_dir).parent / "envs"


def _config() -> Any:
    """Server config when running inside the server, else loaded fresh."""
    try:
        from strata.server import get_state

        return get_state().config
    except RuntimeError:
        from strata.config import StrataConfig

        return StrataConfig.load()
