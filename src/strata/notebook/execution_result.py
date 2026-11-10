"""What running a cell returns, whichever path ran it."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from strata.notebook.immutability import MutationWarning
from strata.notebook.read_audit import ReadFinding


@dataclass(frozen=True)
class CachedConsole:
    """A run's console as its record holds it, with what that run cost."""

    stdout: str
    stderr: str
    run_ms: int = 0


def hit_console(console: CachedConsole | None, *, shows_this_key: bool) -> dict[str, Any]:
    """A hit's console fields on `CellExecutionResult`.

    With no record (a team pull, or a store from before every cell kept one) a console
    from this key stays and one from another key goes: nothing here says what this run
    printed, and the other key's output would read as if it did.
    """
    if console is None:
        return {"replaces_console": not shows_this_key}
    return {
        "stdout": console.stdout,
        "stderr": console.stderr,
        "replaces_console": True,
        "hit_run_ms": console.run_ms,
    }


@dataclass(kw_only=True)
class CellExecutionResult:
    """Result from executing a cell.

    ``display_output`` is a legacy shim for the last of ``display_outputs``;
    the two are kept consistent. ``execution_method`` is ``cold``, ``warm`` or ``cached``.
    """

    cell_id: str
    success: bool
    stdout: str = ""
    stderr: str = ""
    outputs: dict[str, Any] = field(default_factory=dict)
    display_outputs: list[dict[str, Any]] = field(default_factory=list)
    display_output: dict[str, Any] | None = None
    duration_ms: float = 0
    error: str | None = None
    # A stable name for the failure a client can match on (e.g. ``fetch_pin_mismatch``);
    # ``None`` for failures that have none.
    error_code: str | None = None
    # The harness's formatted traceback, kept apart from ``error`` so the UI's one-line
    # pill stays one line.
    traceback: str | None = None
    # With the harness frames that lead to the cell, which ``traceback`` leaves out.
    traceback_full: str | None = None
    cache_hit: bool = False
    # A local mount's file (``mount/path``) that changed while the run read it: its
    # outputs are kept for the run only (``mounts.moved_file``).
    moved_input: str | None = None
    artifact_uri: str | None = None
    execution_method: str = "cold"  # cold, warm, cached
    mutation_warnings: list[MutationWarning] = field(default_factory=list)
    # What the run read that the cell does not declare (``read_audit.report``).
    read_audit: list[ReadFinding] = field(default_factory=list)
    # Retries the prompt-cell validate-and-retry loop used; 0 on first-try pass or for
    # non-schema cells.
    validation_retries: int = 0
    suggest_install: str | None = None  # e.g. "requests" (Python) or "arrow" (R)
    # Frontend dispatches ``uv add`` for ``"python"``, ``install.packages()`` for ``"r"``.
    # ``None`` when ``suggest_install`` is ``None``.
    suggest_install_language: str | None = None
    remote_worker: str | None = None
    remote_transport: str | None = None
    remote_build_id: str | None = None
    remote_build_state: str | None = None
    remote_error_code: str | None = None
    # Explicit: none of the fields below reliably implies a team hit (an unauthenticated
    # store publishes anonymously; a publisher may record no duration).
    from_team_cache: bool = False
    # Who computed a team-store hit. ``None`` for a local hit or a real run; otherwise a
    # team hit with no author reads as a bug, not a saving.
    team_cache_principal: str | None = None
    # Platform the team hit was built on, e.g. ``cpython-3.14-linux-x86_64``. Provenance
    # covers the lockfile, not the platform, so a hit can cross machines; this discloses
    # which. Empty unless a team hit recorded one.
    team_cache_build_env: str = ""
    # The publisher's run cost, i.e. what this hit saved. Carried, not inferred: savings
    # are priced against the last local run, which a teammate-served cell never had.
    team_cache_saved_ms: int = 0
    # The promotion the team hit came from, when one put it in the store.
    team_cache_promotion: str | None = None
    # A hit's stdout and stderr replace the console the cell shows, even when empty.
    replaces_console: bool = False
    # What the run a local hit stands for cost. 0 when its record does not say.
    hit_run_ms: int = 0

    def __post_init__(self) -> None:
        # Legacy shim: accept either `display_outputs` or `display_output` and keep both
        # consistent.
        if not self.display_outputs and self.display_output is not None:
            self.display_outputs = [self.display_output]
        elif self.display_output is None and self.display_outputs:
            self.display_output = self.display_outputs[-1]

    def apply_remote_metadata(
        self,
        *,
        remote_worker: str | None = None,
        remote_transport: str | None = None,
        remote_build_id: str | None = None,
        remote_build_state: str | None = None,
        remote_error_code: str | None = None,
    ) -> CellExecutionResult:
        """Attach remote execution metadata to this result."""
        if remote_worker:
            self.remote_worker = remote_worker
        if remote_transport:
            self.remote_transport = remote_transport
        if remote_build_id:
            self.remote_build_id = remote_build_id
        if remote_build_state:
            self.remote_build_state = remote_build_state
        if remote_error_code:
            self.remote_error_code = remote_error_code
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to the REST ``/agent`` execution-result wire shape.

        The WebSocket frame uses a different, slimmer shape
        (``_execution_result_payload`` in ws.py).
        """
        payload = asdict(self)
        # Wire rename: success: bool -> status: "ready" | "error".
        payload["status"] = "ready" if payload.pop("success") else "error"
        payload["displays"] = payload.pop("display_outputs")
        payload["display"] = payload.pop("display_output")
        # Optional metadata fields are omitted from the wire when falsy.
        for opt_field in (
            "error_code",
            "validation_retries",
            "suggest_install",
            "suggest_install_language",
            "remote_worker",
            "remote_transport",
            "remote_build_id",
            "remote_build_state",
            "remote_error_code",
        ):
            if not payload[opt_field]:
                payload.pop(opt_field)
        return payload
