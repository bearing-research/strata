"""The latency script runs end to end and reports what the phase-2 goals read."""

from __future__ import annotations

from evals.agent_notebook import latency


def test_it_times_the_first_call_and_each_cell_both_ways():
    report = latency.measure(calls=2, cold_uv_cache=False)

    mcp = report["mcp"]
    assert list(mcp["setup_ms"]) == [
        "creating the scratchpad",
        "starting the scratchpad",
        "opening the scratchpad",
        "syncing the scratchpad's environment",
    ]
    assert mcp["first_cell_ms"] == mcp["first_call_ms"] - sum(mcp["setup_ms"].values())
    for way in (mcp, report["cli"]):
        assert way["uncached"]["calls"] == 2
        assert way["uncached"]["p50_ms"] <= way["uncached"]["p95_ms"] <= way["uncached"]["max_ms"]
