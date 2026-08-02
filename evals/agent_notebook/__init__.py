"""Agent-notebook eval suite.

Measures whether a coding agent, handed a Strata notebook via the
`strata agent` on-ramp, actually *drives the notebook* through the MCP tools —
or routes around it with scratch Python. The headline metric is the **in-tool
rate**; task completion and efficiency are reported alongside.

The graders consume a driver-agnostic :class:`~evals.agent_notebook.trajectory.Trajectory`,
so the same scoring runs against a real Claude Code headless session (local,
on-demand) and against recorded transcripts (deterministic, CI).
"""
