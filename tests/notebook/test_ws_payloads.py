"""Tests for the typed WS frame payload models (#44)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from strata.notebook.models import CellStatus, CellTestCase
from strata.notebook.ws_payloads import (
    CascadeProgressPayload,
    CascadePromptPayload,
    CellConsolePayload,
    CellIterationProgressPayload,
    CellOutputDeltaPayload,
    CellStatusPayload,
    CellTestResultsPayload,
    CellTestStatusPayload,
    EnvironmentJobEventPayload,
    cell_status_payload,
    environment_job_event_payload,
    impact_preview_payload,
    profiling_summary_payload,
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


def test_cell_status_simple_omits_optional_fields():
    # A bare status change must serialize to exactly {cell_id, status} —
    # the optional remote/staleness fields are dropped, preserving the
    # historical wire shape of the many simple emit sites.
    wire = cell_status_payload("c1", "idle")
    assert wire == {"cell_id": "c1", "status": "idle"}


def test_cell_status_coerces_enum_status():
    # Emit sites pass either a CellStatus enum or a plain string.
    wire = cell_status_payload("c1", CellStatus.ERROR)
    assert wire == {"cell_id": "c1", "status": "error"}


def test_cell_status_running_includes_remote_fields():
    wire = cell_status_payload("c1", "running", remote_worker="gpu-box", remote_transport="signed")
    assert wire == {
        "cell_id": "c1",
        "status": "running",
        "remote_worker": "gpu-box",
        "remote_transport": "signed",
    }


def test_cell_status_staleness_keeps_empty_reasons_list():
    # The staleness builder always emits staleness_reasons, even when empty —
    # exclude_none drops None but keeps an empty list, matching prior behavior.
    wire = cell_status_payload("c1", "stale", staleness_reasons=[])
    assert wire == {"cell_id": "c1", "status": "stale", "staleness_reasons": []}


def test_cell_status_staleness_with_reasons_and_causality():
    wire = cell_status_payload(
        "c1",
        "stale",
        staleness_reasons=["upstream_changed"],
        causality={"reason": "upstream", "details": []},
    )
    assert wire == {
        "cell_id": "c1",
        "status": "stale",
        "staleness_reasons": ["upstream_changed"],
        "causality": {"reason": "upstream", "details": []},
    }


def test_cell_status_forbids_extra_field():
    with pytest.raises(ValidationError):
        CellStatusPayload(cell_id="c1", status="idle", bogus=1)


def test_cascade_prompt_payload():
    wire = CascadePromptPayload(
        cell_id="c3", plan_id="p1", cells_to_run=["c1", "c2"], estimated_duration_ms=120
    ).model_dump(mode="json")
    assert wire == {
        "cell_id": "c3",
        "plan_id": "p1",
        "cells_to_run": ["c1", "c2"],
        "estimated_duration_ms": 120,
    }


def test_cascade_progress_payload():
    wire = CascadeProgressPayload(
        plan_id="p1", current_cell_id="c2", completed=1, total=3
    ).model_dump(mode="json")
    assert wire == {"plan_id": "p1", "current_cell_id": "c2", "completed": 1, "total": 3}


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


def test_environment_job_event_wraps_snapshot():
    job = {
        "id": "j1",
        "action": "add",
        "command": "uv add numpy",
        "status": "running",
        "started_at": 1000,
        "package": "numpy",
        "phase": "resolving",
        "stale_cell_ids": ["c1"],
    }
    wire = environment_job_event_payload(job)
    inner = wire["environment_job"]
    assert inner["id"] == "j1"
    assert inner["package"] == "numpy"
    assert inner["stale_cell_ids"] == ["c1"]
    # Snapshot defaults fill in for the omitted fields.
    assert inner["stdout"] == ""
    assert inner["lockfile_changed"] is False
    assert inner["finished_at"] is None


def test_impact_preview_validates_nested_steps():
    from dataclasses import asdict

    from strata.notebook.cascade import CascadeReason, CascadeStep
    from strata.notebook.impact import DownstreamImpact, ImpactPreview

    impact = ImpactPreview(
        target_cell_id="c2",
        upstream=[CascadeStep(cell_id="c1", cell_name="load", reason=CascadeReason.STALE)],
        downstream=[
            DownstreamImpact(cell_id="c3", cell_name="plot", current_status="ready"),
        ],
        estimated_ms=42,
    )
    wire = impact_preview_payload(asdict(impact))
    assert wire["target_cell_id"] == "c2"
    assert wire["estimated_ms"] == 42
    assert wire["upstream"][0] == {
        "cell_id": "c1",
        "cell_name": "load",
        "reason": "stale",  # CascadeReason StrEnum → its value
        "skip": False,
        "estimated_ms": 0,
    }
    assert wire["downstream"][0]["new_status"] == "stale:upstream"


def test_profiling_summary_validates_and_coerces_status_enum():
    from strata.notebook.models import CellStatus

    summary = {
        "total_execution_ms": 100,
        "cache_hits": 2,
        "cache_misses": 1,
        "cache_savings_ms": 30,
        "total_artifact_bytes": 4096,
        "cell_profiles": [
            {
                "cell_id": "c1",
                "cell_name": "x",
                "status": CellStatus.READY,  # enum, as get_profiling_summary emits it
                "duration_ms": 12,
                "cache_hit": True,
                "artifact_uri": None,
                "execution_count": 3,
            }
        ],
    }
    wire = profiling_summary_payload(summary)
    assert wire["total_execution_ms"] == 100
    assert wire["cell_profiles"][0]["status"] == "ready"
    assert wire["cell_profiles"][0]["artifact_uri"] is None


def test_environment_job_model_matches_snapshot_fields():
    # Drift guard: the typed model must mirror the EnvironmentJobSnapshot
    # dataclass exactly. asdict(job) is validated with extra="forbid", so a new
    # snapshot field (or a removed model field) fails loudly here instead of
    # silently changing the wire contract.
    import dataclasses

    from strata.notebook.session import EnvironmentJobSnapshot

    snapshot_fields = {f.name for f in dataclasses.fields(EnvironmentJobSnapshot)}
    model_fields = set(
        EnvironmentJobEventPayload.model_fields["environment_job"].annotation.model_fields
    )
    assert model_fields == snapshot_fields


def test_dag_update_payload_round_trips_realistic_shape():
    """A realistic dag_update — edges, topology, a module cell + a plain cell,
    and a variant group — validates and round-trips without dropping fields."""
    from strata.notebook.ws_payloads import dag_update_payload

    raw = {
        "edges": [
            {"from_cell_id": "a", "to_cell_id": "b", "variable": "x"},
        ],
        "roots": ["a"],
        "leaves": ["b"],
        "topological_order": ["a", "b"],
        "cells": [
            {
                "id": "a",
                "defines": ["featurize"],
                "references": [],
                "upstream_ids": [],
                "downstream_ids": ["b"],
                "is_leaf": False,
                "annotation_diagnostics": [],
                "variant_group": None,
                "variant_name": None,
                "variant_active": None,
                "is_module_cell": True,
                "module_exports": [{"name": "featurize", "kind": "function"}],
            },
            {
                # A non-Python / non-module cell omits is_module_cell + module_exports.
                "id": "b",
                "defines": ["y"],
                "references": ["x"],
                "upstream_ids": ["a"],
                "downstream_ids": [],
                "is_leaf": True,
                "annotation_diagnostics": [{"code": "worker_unknown", "message": "…"}],
                "variant_group": "model",
                "variant_name": "rf",
                "variant_active": True,
            },
        ],
        "variant_groups": [{"group": "model", "active": "rf", "members": ["rf", "xgb"]}],
    }

    wire = dag_update_payload(raw)

    assert [e["from_cell_id"] for e in wire["edges"]] == ["a"]
    assert wire["topological_order"] == ["a", "b"]
    module_cell = next(c for c in wire["cells"] if c["id"] == "a")
    assert module_cell["is_module_cell"] is True
    assert module_cell["module_exports"] == [{"name": "featurize", "kind": "function"}]
    plain_cell = next(c for c in wire["cells"] if c["id"] == "b")
    assert plain_cell["is_module_cell"] is False  # default when omitted
    assert plain_cell["module_exports"] is None
    assert plain_cell["annotation_diagnostics"][0]["code"] == "worker_unknown"
    assert wire["variant_groups"][0]["group"] == "model"


def test_dag_update_payload_rejects_unmodeled_cell_field():
    from strata.notebook.ws_payloads import dag_update_payload

    with pytest.raises(ValidationError):
        dag_update_payload(
            {
                "cells": [
                    {"id": "a", "is_leaf": True, "surprise_field": 1},
                ],
            }
        )
