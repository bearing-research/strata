"""Tests for the typed WS frame payload models (#44)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from strata.notebook.models import CellTestCase
from strata.notebook.ws_payloads import (
    CellConsolePayload,
    CellIterationProgressPayload,
    CellOutputDeltaPayload,
    CellTestResultsPayload,
    CellTestStatusPayload,
)


def test_console_payload_roundtrips_to_wire_dict():
    wire = CellConsolePayload(cell_id="c1", stream="stdout", text="hi").model_dump(mode="json")
    assert wire == {"cell_id": "c1", "stream": "stdout", "text": "hi"}


def test_console_rejects_unknown_stream():
    with pytest.raises(ValidationError):
        CellConsolePayload(cell_id="c1", stream="stdlog", text="x")


def test_extra_field_is_forbidden():
    # The whole point of typing the protocol: an unmodeled field is a loud error
    # at the boundary, not a silently-shipped wire field.
    with pytest.raises(ValidationError):
        CellConsolePayload(cell_id="c1", stream="stdout", text="x", bogus=1)


def test_output_delta_kinds():
    for kind in ("delta", "retry", "notice"):
        wire = CellOutputDeltaPayload(cell_id="c1", attempt=1, kind=kind, text="t").model_dump(
            mode="json"
        )
        assert wire["kind"] == kind
    with pytest.raises(ValidationError):
        CellOutputDeltaPayload(cell_id="c1", attempt=1, kind="bogus", text="t")


def test_iteration_progress_defaults_and_validation():
    wire = CellIterationProgressPayload(
        cell_id="c1", iteration=2, max_iter=10, duration_ms=5
    ).model_dump(mode="json")
    assert wire["artifact_uri"] is None
    assert wire["until_reached"] is False
    # max_iter is required.
    with pytest.raises(ValidationError):
        CellIterationProgressPayload(cell_id="c1", iteration=2, duration_ms=5)


def test_test_status_payload():
    wire = CellTestStatusPayload(cell_id="c1", status="running").model_dump(mode="json")
    assert wire == {"cell_id": "c1", "status": "running"}
    with pytest.raises(ValidationError):
        CellTestStatusPayload(cell_id="c1", status="pending")


def test_test_results_serializes_nested_cases_and_drops_internal_hashes():
    payload = CellTestResultsPayload(
        cell_id="c1",
        passed=1,
        failed=1,
        errored=0,
        skipped=0,
        tests=[
            CellTestCase(name="t_ok", nodeid="test_cell.py::t_ok", outcome="passed", message=""),
            CellTestCase(
                name="t_bad",
                nodeid="test_cell.py::t_bad",
                outcome="failed",
                message="assert 3 == 5",
            ),
        ],
        stale=False,
        pytest_unavailable=False,
        ran_at=123,
    )
    wire = payload.model_dump(mode="json")
    # Internal staleness hashes are not on the wire (they were a model_dump leak).
    assert "cell_source_hash" not in wire
    assert "test_source_hash" not in wire
    assert "input_fingerprint" not in wire
    # Client-facing fields + nested cases serialize as plain dicts.
    assert wire["passed"] == 1
    assert wire["tests"][1]["message"] == "assert 3 == 5"
    assert set(wire) == {
        "cell_id",
        "passed",
        "failed",
        "errored",
        "skipped",
        "tests",
        "stale",
        "pytest_unavailable",
        "ran_at",
    }
