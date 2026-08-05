"""Graders over a normalized :class:`Trajectory` and the final notebook.

All graders are **pure**: they take a trajectory (and, for completion, a
notebook directory + a precomputed ``run_ok`` flag) and return a dataclass.
Nothing here spawns a server, an LLM, or a subprocess — so the CI replay test
scores recorded runs with neither a venv nor network.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from .trajectory import Trajectory


@dataclass
class InToolGrade:
    """The headline adoption metric: work done in the notebook vs. routed around it."""

    notebook_work: int
    escapes: int
    in_tool_rate: float
    no_activity: bool
    escape_details: list[str] = field(default_factory=list)


def grade_in_tool(traj: Trajectory) -> InToolGrade:
    """Fraction of work actions taken through the notebook's MCP tools.

    ``in_tool_rate = notebook_work / (notebook_work + escapes)``, where an escape
    is any bypass of the notebook — running Python via Bash, installing packages
    via Bash, or editing a cell file directly (see ``ToolEvent.escape_reason``).
    A run that did neither is ``no_activity`` (rate reported as 1.0 but excluded
    from suite aggregates — it's a completion failure, not an adoption signal).
    """
    work = traj.work_events()
    escapes = traj.escape_events()
    denom = len(work) + len(escapes)
    no_activity = denom == 0
    rate = 1.0 if no_activity else len(work) / denom
    return InToolGrade(
        notebook_work=len(work),
        escapes=len(escapes),
        in_tool_rate=rate,
        no_activity=no_activity,
        escape_details=[f"{e.escape_reason}: {e.escape_detail()}" for e in escapes],
    )


@dataclass
class CompletionGrade:
    """Did the agent leave a correct, runnable notebook?"""

    runs_clean: bool | None
    missing_variables: list[str]
    passed: bool


def notebook_defines(notebook_dir: Path) -> set[str]:
    """Every top-level variable defined across the notebook's cells.

    Runs the same static analysis the DAG uses (``analyze_cell``) over each
    cell's source — ``parse_notebook`` alone doesn't populate ``cell.defines``,
    that happens during variable analysis.
    """
    from strata.notebook.analyzer import analyze_cell
    from strata.notebook.parser import parse_notebook

    state = parse_notebook(notebook_dir)
    defined: set[str] = set()
    for cell in state.cells:
        defined.update(analyze_cell(cell.source).defines)
    return defined


def grade_completion(
    notebook_dir: Path, expect_variables: list[str], run_ok: bool | None
) -> CompletionGrade:
    """Score the final notebook against a task's expectations.

    ``run_ok`` is supplied by the caller (the runner shells out to ``strata
    run``); pass ``None`` when a runnable check was not performed. The grader
    itself stays pure so CI can score a fixture notebook without a venv.
    """
    defined = notebook_defines(notebook_dir)
    missing = [v for v in expect_variables if v not in defined]
    passed = not missing and run_ok is not False
    return CompletionGrade(runs_clean=run_ok, missing_variables=missing, passed=passed)


@dataclass
class EfficiencyGrade:
    """How much the agent did, so a high in-tool rate isn't confused with churn."""

    tool_calls: int
    notebook_reads: int
    notebook_work: int
    escapes: int


def grade_efficiency(traj: Trajectory) -> EfficiencyGrade:
    return EfficiencyGrade(
        tool_calls=traj.tool_calls,
        notebook_reads=sum(1 for e in traj.events if e.is_notebook_read),
        notebook_work=len(traj.work_events()),
        escapes=len(traj.escape_events()),
    )


@dataclass
class RunResult:
    """The scored outcome of one task run, ready to serialize into a report."""

    task_id: str
    driver: str
    in_tool: InToolGrade
    completion: CompletionGrade
    efficiency: EfficiencyGrade
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "driver": self.driver,
            "error": self.error,
            "in_tool": asdict(self.in_tool),
            "completion": asdict(self.completion),
            "efficiency": asdict(self.efficiency),
        }


def score_run(
    task_id: str,
    driver: str,
    traj: Trajectory,
    notebook_dir: Path,
    expect_variables: list[str],
    run_ok: bool | None,
    error: str | None = None,
) -> RunResult:
    """Apply every grader and assemble a :class:`RunResult`."""
    return RunResult(
        task_id=task_id,
        driver=driver,
        in_tool=grade_in_tool(traj),
        completion=grade_completion(notebook_dir, expect_variables, run_ok),
        efficiency=grade_efficiency(traj),
        error=error,
    )
