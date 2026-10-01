"""Agent-notebook eval suite.

Measures whether a coding agent handed a notebook via `strata agent` drives it
through the MCP tools or routes around it with scratch Python. The headline
metric is the **in-tool rate**. Graders score a driver-agnostic
:class:`~evals.agent_notebook.trajectory.Trajectory`, so live runs and recorded
transcripts (CI) share one scoring path.
"""
