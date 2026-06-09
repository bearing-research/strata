"""Unit tests for ``harness.execute_batch`` — the library function that
drives a sequence of cells in one Python process.

These tests drive the function from a background thread and act as the
"fake parent" on the other end of two real ``os.pipe`` pairs. No
subprocess yet; that's PR-b2's job.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import orjson
import pytest

from strata.notebook.harness import execute_batch

# ---------------------------------------------------------------------------
# Pipe / thread plumbing
# ---------------------------------------------------------------------------


def _read_frame(stream: Any) -> dict | None:
    """Read one length-line JSON frame from the harness."""
    line = stream.readline()
    if not line:
        return None
    return orjson.loads(line)


def _send_response(stream: Any, payload: dict) -> None:
    """Write one JSON response line to the harness."""
    stream.write(orjson.dumps(payload) + b"\n")
    stream.flush()


@pytest.fixture
def batch_pipes(tmp_path):
    """Yields (frame_r, frame_w, resp_r, resp_w, output_dir) — two pipe
    pairs wrapped as buffered file objects plus a temp output dir.
    """
    frame_r, frame_w = os.pipe()
    resp_r, resp_w = os.pipe()
    output_dir = tmp_path / "batch_out"
    output_dir.mkdir()

    frame_w_f = os.fdopen(frame_w, "wb")
    resp_r_f = os.fdopen(resp_r, "rb")
    frame_r_f = os.fdopen(frame_r, "rb")
    resp_w_f = os.fdopen(resp_w, "wb")

    yield frame_r_f, frame_w_f, resp_r_f, resp_w_f, output_dir

    for stream in (frame_w_f, resp_r_f, frame_r_f, resp_w_f):
        try:
            stream.close()
        except Exception:
            pass


def _run_in_thread(
    cells: list[dict],
    upstream_inputs: dict,
    output_dir: Path,
    frame_w_f: Any,
    resp_r_f: Any,
) -> tuple[threading.Thread, list[BaseException]]:
    errors: list[BaseException] = []

    def target() -> None:
        try:
            execute_batch(cells, upstream_inputs, output_dir, frame_w_f, resp_r_f)
        except BaseException as exc:
            errors.append(exc)
        finally:
            # Close the write end so the test thread's readline() returns
            # EOF once frames are drained.
            try:
                frame_w_f.close()
            except Exception:
                pass

    thread = threading.Thread(target=target)
    thread.start()
    return thread, errors


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_batch_runs_two_cells_with_cache_miss_then_persist(batch_pipes):
    """Two cells, both cache-miss, both succeed. Verify frame sequence
    and that serialized blobs land at output_dir/<cell_id>/{var}{ext}.
    """
    frame_r, frame_w, resp_r, resp_w, output_dir = batch_pipes

    cells = [
        {
            "cell_id": "c1",
            "source": "x = 41",
            "consumed_vars": ["x"],
            "env": {},
            "mount_manifest": {},
            "source_hash": "src-c1",
            "env_hash": "env",
        },
        {
            "cell_id": "c2",
            "source": "y = x + 1",
            "consumed_vars": ["y"],
            "env": {},
            "mount_manifest": {},
            "source_hash": "src-c2",
            "env_hash": "env",
        },
    ]

    thread, errors = _run_in_thread(cells, {}, output_dir, frame_w, resp_r)

    frames: list[dict] = []
    while True:
        frame = _read_frame(frame_r)
        if frame is None:
            break
        frames.append(frame)
        if frame["type"] == "cache_check":
            _send_response(resp_w, {"cache_hit": False, "provenance_hash": "abc"})
        elif frame["type"] == "persist":
            _send_response(
                resp_w, {"ok": True, "uri": f"strata://test/{frame['payload']['cell_id']}"}
            )
        elif frame["type"] == "batch_end":
            break

    thread.join(timeout=5)
    assert not thread.is_alive(), "harness thread did not exit"
    assert not errors, f"harness raised: {errors!r}"

    types = [f["type"] for f in frames]
    assert types == [
        "cell_start",
        "cache_check",
        "persist",
        "cell_start",
        "cache_check",
        "persist",
        "batch_end",
    ], f"unexpected frame sequence: {types}"

    # Persist payloads carry the outputs.
    persist_c1 = next(
        f["payload"] for f in frames if f["type"] == "persist" and f["payload"]["cell_id"] == "c1"
    )
    assert "x" in persist_c1["outputs"]
    assert persist_c1["outputs"]["x"]["preview"] == 41

    persist_c2 = next(
        f["payload"] for f in frames if f["type"] == "persist" and f["payload"]["cell_id"] == "c2"
    )
    assert "y" in persist_c2["outputs"]
    assert persist_c2["outputs"]["y"]["preview"] == 42

    # Files on disk under per-cell subdirs.
    assert (output_dir / "c1" / persist_c1["outputs"]["x"]["file"]).exists()
    assert (output_dir / "c2" / persist_c2["outputs"]["y"]["file"]).exists()

    end_frame = frames[-1]
    assert end_frame["type"] == "batch_end"
    assert end_frame["payload"]["reason"] == "complete"


# ---------------------------------------------------------------------------
# Cell error mid-batch
# ---------------------------------------------------------------------------


def test_batch_ends_on_cell_error(batch_pipes):
    """Second cell raises; batch ends with cell_error reason and the
    third cell never starts.
    """
    frame_r, frame_w, resp_r, resp_w, output_dir = batch_pipes

    cells = [
        {
            "cell_id": "c1",
            "source": "x = 1",
            "consumed_vars": ["x"],
            "env": {},
            "mount_manifest": {},
        },
        {
            "cell_id": "c2",
            "source": "raise RuntimeError('boom')",
            "consumed_vars": [],
            "env": {},
            "mount_manifest": {},
        },
        {
            "cell_id": "c3",
            "source": "z = 99",
            "consumed_vars": ["z"],
            "env": {},
            "mount_manifest": {},
        },
    ]

    thread, errors = _run_in_thread(cells, {}, output_dir, frame_w, resp_r)

    frames: list[dict] = []
    while True:
        frame = _read_frame(frame_r)
        if frame is None:
            break
        frames.append(frame)
        if frame["type"] == "cache_check":
            _send_response(resp_w, {"cache_hit": False, "provenance_hash": "abc"})
        elif frame["type"] == "persist":
            _send_response(resp_w, {"ok": True, "uri": "strata://test"})
        elif frame["type"] == "batch_end":
            break

    thread.join(timeout=5)
    assert not errors

    types = [f["type"] for f in frames]
    # c1 succeeds; c2 errors; c3 must NOT have a cell_start.
    assert "cell_start" in types
    assert types.count("cell_start") == 2, f"c3 must not start: {types}"
    cell_errors = [f for f in frames if f["type"] == "cell_error"]
    assert len(cell_errors) == 1
    assert cell_errors[0]["payload"]["cell_id"] == "c2"
    assert "RuntimeError" in cell_errors[0]["payload"]["traceback"]

    end_frame = frames[-1]
    assert end_frame["type"] == "batch_end"
    assert end_frame["payload"]["reason"] == "cell_error"
    assert end_frame["payload"]["failed_cell_id"] == "c2"


# ---------------------------------------------------------------------------
# Cache hit
# ---------------------------------------------------------------------------


def test_batch_cache_hit_loads_from_disk_and_continues(batch_pipes, tmp_path):
    """Cache hit on c1: parent has materialized the blob; harness loads
    it into the namespace and c2 sees the value.
    """
    frame_r, frame_w, resp_r, resp_w, output_dir = batch_pipes

    # Pre-write a cached blob for c1.x as if the parent had materialized it
    # from the artifact store.
    cached_x_path = output_dir / "c1" / "x.json"
    cached_x_path.parent.mkdir(parents=True, exist_ok=True)
    cached_x_path.write_bytes(orjson.dumps(7))

    cells = [
        {
            "cell_id": "c1",
            "source": "x = 41",  # Source differs from cached value to
            # verify we LOAD the cached value, not execute.
            "consumed_vars": ["x"],
            "env": {},
            "mount_manifest": {},
        },
        {
            "cell_id": "c2",
            "source": "y = x * 2",
            "consumed_vars": ["y"],
            "env": {},
            "mount_manifest": {},
        },
    ]

    thread, errors = _run_in_thread(cells, {}, output_dir, frame_w, resp_r)

    frames: list[dict] = []
    while True:
        frame = _read_frame(frame_r)
        if frame is None:
            break
        frames.append(frame)
        if frame["type"] == "cache_check":
            cell_id = frame["payload"]["cell_id"]
            if cell_id == "c1":
                _send_response(
                    resp_w,
                    {
                        "cache_hit": True,
                        "provenance_hash": "p1",
                        "cached_outputs": {"x": {"content_type": "json/object", "file": "x.json"}},
                        "cached_displays": [],
                    },
                )
            else:
                _send_response(resp_w, {"cache_hit": False, "provenance_hash": "p2"})
        elif frame["type"] == "persist":
            _send_response(resp_w, {"ok": True, "uri": "strata://test"})
        elif frame["type"] == "batch_end":
            break

    thread.join(timeout=5)
    assert not errors

    persist_c2 = next(
        f["payload"] for f in frames if f["type"] == "persist" and f["payload"]["cell_id"] == "c2"
    )
    # c2 used the CACHED x=7, so y = 14 — proves cache load worked.
    assert persist_c2["outputs"]["y"]["preview"] == 14


# ---------------------------------------------------------------------------
# Mount name save/restore
# ---------------------------------------------------------------------------


def test_display_capture_reinstalls_per_cell(batch_pipes):
    """Each cell needs its own DisplayCapture handler. ``install()`` uses
    ``setdefault`` so once the namespace has a ``display`` key from cell A,
    cell B's calls to ``display(...)`` would go to A's capture (and into
    cell A's display values). Verify cell B's display captures cell B's
    payload, not cell A's.
    """
    frame_r, frame_w, resp_r, resp_w, output_dir = batch_pipes

    cells = [
        # ``display`` and ``Markdown`` are injected into the cell namespace
        # by ``DisplayCapture.install`` — no import needed.
        {
            "cell_id": "c1",
            "source": "display(Markdown('cell A'))\n",
            "consumed_vars": [],
            "env": {},
            "mount_manifest": {},
        },
        {
            "cell_id": "c2",
            "source": "display(Markdown('cell B'))\n",
            "consumed_vars": [],
            "env": {},
            "mount_manifest": {},
        },
    ]

    thread, errors = _run_in_thread(cells, {}, output_dir, frame_w, resp_r)

    frames: list[dict] = []
    while True:
        frame = _read_frame(frame_r)
        if frame is None:
            break
        frames.append(frame)
        if frame["type"] == "cache_check":
            _send_response(resp_w, {"cache_hit": False, "provenance_hash": "p"})
        elif frame["type"] == "persist":
            _send_response(resp_w, {"ok": True, "uri": "strata://test"})
        elif frame["type"] == "batch_end":
            break

    thread.join(timeout=5)
    assert not errors

    persists = {p["payload"]["cell_id"]: p["payload"] for p in frames if p["type"] == "persist"}
    # Each cell should have produced exactly one display output. Without the
    # fix, cell B's display call would have hit cell A's now-orphaned capture
    # and cell B's display_outputs would be empty.
    assert len(persists["c1"]["display_outputs"]) == 1, persists["c1"]["display_outputs"]
    assert len(persists["c2"]["display_outputs"]) == 1, persists["c2"]["display_outputs"]


def test_display_filenames_use_serializer_convention(batch_pipes):
    """Harness serializes display outputs as ``__display__N{ext}``, the
    naming convention the serializer's content-type detection recognizes
    (``_is_display_variable_name`` in serializer.py L312). With ``display_N``
    the values would be classified as regular pickles.
    """
    frame_r, frame_w, resp_r, resp_w, output_dir = batch_pipes

    cells = [
        {
            "cell_id": "c1",
            "source": "display(Markdown('hello'))\n",
            "consumed_vars": [],
            "env": {},
            "mount_manifest": {},
        },
    ]

    thread, errors = _run_in_thread(cells, {}, output_dir, frame_w, resp_r)

    frames: list[dict] = []
    while True:
        frame = _read_frame(frame_r)
        if frame is None:
            break
        frames.append(frame)
        if frame["type"] == "cache_check":
            _send_response(resp_w, {"cache_hit": False, "provenance_hash": "p"})
        elif frame["type"] == "persist":
            _send_response(resp_w, {"ok": True, "uri": "strata://test"})
        elif frame["type"] == "batch_end":
            break

    thread.join(timeout=5)
    assert not errors

    persist = next(p["payload"] for p in frames if p["type"] == "persist")
    assert persist["display_outputs"], "expected at least one display output"
    display_meta = persist["display_outputs"][0]
    # Filename must start with __display__ so the serializer treats it
    # as display content (not a regular pickle).
    assert display_meta["file"].startswith("__display__"), display_meta


def test_mount_name_save_restore(batch_pipes, tmp_path):
    """A cell-level mount must not clobber a pre-existing user variable
    with the same name. After the mount-declaring cell, the user value
    is restored.
    """
    frame_r, frame_w, resp_r, resp_w, output_dir = batch_pipes

    mount_dir = tmp_path / "mounted"
    mount_dir.mkdir()

    cells = [
        {
            "cell_id": "c1",
            "source": "data = 'user value'",
            "consumed_vars": ["data"],
            "env": {},
            "mount_manifest": {},
        },
        {
            "cell_id": "c2",
            "source": "from pathlib import Path; saw_path = isinstance(data, Path)",
            "consumed_vars": ["saw_path"],
            "env": {},
            "mount_manifest": {"data": {"local_path": str(mount_dir), "mode": "ro"}},
        },
        {
            "cell_id": "c3",
            "source": "still_user = data",
            "consumed_vars": ["still_user"],
            "env": {},
            "mount_manifest": {},
        },
    ]

    thread, errors = _run_in_thread(cells, {}, output_dir, frame_w, resp_r)

    frames: list[dict] = []
    while True:
        frame = _read_frame(frame_r)
        if frame is None:
            break
        frames.append(frame)
        if frame["type"] == "cache_check":
            _send_response(resp_w, {"cache_hit": False, "provenance_hash": "p"})
        elif frame["type"] == "persist":
            _send_response(resp_w, {"ok": True, "uri": "strata://test"})
        elif frame["type"] == "batch_end":
            break

    thread.join(timeout=5)
    assert not errors

    # c2 saw `data` as a Path (the mount binding).
    persist_c2 = next(
        f["payload"] for f in frames if f["type"] == "persist" and f["payload"]["cell_id"] == "c2"
    )
    assert persist_c2["outputs"]["saw_path"]["preview"] is True

    # c3 saw `data` restored to "user value" — mount didn't leak.
    persist_c3 = next(
        f["payload"] for f in frames if f["type"] == "persist" and f["payload"]["cell_id"] == "c3"
    )
    assert persist_c3["outputs"]["still_user"]["preview"] == "user value"


# ---------------------------------------------------------------------------
# R-only artifact (RDS) seeding
# ---------------------------------------------------------------------------


def test_batch_surfaces_rds_input_as_first_consumer_cell_error(batch_pipes):
    """An RDS upstream consumed by a Python cell becomes a cell_error
    on the first consumer — not subprocess_died.

    Pre-fix behaviour: ``deserialize_inputs`` raised ``StrataRArtifactError``
    inside ``execute_batch`` before any ``cell_start`` frame fired, so the
    parent's frame-reader hit EOF and reported the batch as
    ``subprocess_died``. The fix defers the error to the first cell whose
    source references the tainted variable, emitting a proper
    ``cell_start`` + ``cell_error`` + ``batch_end`` sequence with the
    structured "re-export as data.frame" message.
    """
    frame_r, frame_w, resp_r, resp_w, output_dir = batch_pipes

    # Drop an RDS blob in the dir the harness reads upstream inputs from.
    # Bytes are irrelevant — the dispatcher rejects on content_type.
    rds_path = output_dir / "fit.rds"
    rds_path.write_bytes(b"\x1f\x8b\x08\x00fakerds")

    upstream_inputs = {
        "fit": {
            "content_type": "application/x-r-rds",
            "file": "fit.rds",
        }
    }

    cells = [
        # c1 doesn't reference `fit` — should run cleanly.
        {
            "cell_id": "c1",
            "source": "x = 1",
            "consumed_vars": ["x"],
            "env": {},
            "mount_manifest": {},
        },
        # c2 references `fit` — first cell that triggers the tainted error.
        {
            "cell_id": "c2",
            "source": "score = fit + 1",
            "consumed_vars": ["score"],
            "env": {},
            "mount_manifest": {},
        },
        # c3 must NOT start once c2 errors.
        {
            "cell_id": "c3",
            "source": "y = 2",
            "consumed_vars": ["y"],
            "env": {},
            "mount_manifest": {},
        },
    ]

    thread, errors = _run_in_thread(cells, upstream_inputs, output_dir, frame_w, resp_r)

    frames: list[dict] = []
    while True:
        frame = _read_frame(frame_r)
        if frame is None:
            break
        frames.append(frame)
        if frame["type"] == "cache_check":
            _send_response(resp_w, {"cache_hit": False, "provenance_hash": "abc"})
        elif frame["type"] == "persist":
            _send_response(resp_w, {"ok": True, "uri": "strata://test"})
        elif frame["type"] == "batch_end":
            break

    thread.join(timeout=5)
    # Critical: no exception escaped — the previous behaviour was a raw
    # propagation that the test thread captured here.
    assert not errors, f"execute_batch let the error escape: {errors}"

    types = [f["type"] for f in frames]
    # c1 ran normally (cell_start + cache_check + persist).
    assert types.count("cell_start") == 2, f"expected only c1+c2 to start, got: {types}"

    cell_errors = [f for f in frames if f["type"] == "cell_error"]
    assert len(cell_errors) == 1
    err_payload = cell_errors[0]["payload"]
    assert err_payload["cell_id"] == "c2"
    # The structured message — variable name + saveRDS + data.frame
    # suggestion — must all be present so the UI surfaces the
    # actionable text rather than NameError.
    assert "fit" in err_payload["error"]
    assert "saveRDS" in err_payload["error"]
    assert "data.frame" in err_payload["error"]

    end_frame = frames[-1]
    assert end_frame["type"] == "batch_end"
    assert end_frame["payload"]["reason"] == "cell_error"
    assert end_frame["payload"]["failed_cell_id"] == "c2"


def test_batch_word_boundary_avoids_spurious_taint(batch_pipes):
    """``fit`` in another identifier (``unfit_data``) or a string literal
    must not trigger the tainted-input branch.

    Without word-boundary matching, a tainted ``fit`` would block any
    cell whose source contained the substring ``fit`` — including
    ``unfit_data = ...``, comments, and string literals like
    ``"benefit"``. Use a clearly substring-only case here so the
    test fails loudly if the matcher regresses to ``var in source``.
    """
    frame_r, frame_w, resp_r, resp_w, output_dir = batch_pipes

    rds_path = output_dir / "fit.rds"
    rds_path.write_bytes(b"rds")

    upstream_inputs = {"fit": {"content_type": "application/x-r-rds", "file": "fit.rds"}}

    cells = [
        {
            "cell_id": "c1",
            # ``unfit_data`` and ``fitness`` contain ``fit`` as a substring
            # but neither is the bare identifier — must NOT trip taint.
            "source": "unfit_data = 1\nfitness = unfit_data + 1",
            "consumed_vars": ["unfit_data", "fitness"],
            "env": {},
            "mount_manifest": {},
        },
    ]

    thread, errors = _run_in_thread(cells, upstream_inputs, output_dir, frame_w, resp_r)

    frames: list[dict] = []
    while True:
        frame = _read_frame(frame_r)
        if frame is None:
            break
        frames.append(frame)
        if frame["type"] == "cache_check":
            _send_response(resp_w, {"cache_hit": False, "provenance_hash": "p"})
        elif frame["type"] == "persist":
            _send_response(resp_w, {"ok": True, "uri": "strata://test"})
        elif frame["type"] == "batch_end":
            break

    thread.join(timeout=5)
    assert not errors

    cell_errors = [f for f in frames if f["type"] == "cell_error"]
    assert not cell_errors, f"taint matcher mis-fired on substring: {cell_errors}"
    end_frame = frames[-1]
    assert end_frame["payload"]["reason"] == "complete"


def test_batch_injects_ambient_client(batch_pipes):
    """A ``strata_url`` in a batched cell injects the ambient ``strata``
    client into the shared namespace — the batch path is a SEPARATE
    injection site from single-cell/warm-pool (the #145 trap), so it must
    be wired too. The client is excluded from outputs and closed on exit.
    """
    frame_r, frame_w, resp_r, resp_w, output_dir = batch_pipes

    cells = [
        {
            "cell_id": "c1",
            # NameError if strata is not injected in the batch path.
            "source": "client_type = type(strata).__name__",
            "consumed_vars": ["client_type"],
            "env": {},
            "mount_manifest": {},
            "table_manifest": {},
            "strata_url": "http://127.0.0.1:8765",
            "source_hash": "src-c1",
            "env_hash": "env",
        },
    ]

    thread, errors = _run_in_thread(cells, {}, output_dir, frame_w, resp_r)

    frames: list[dict] = []
    while True:
        frame = _read_frame(frame_r)
        if frame is None:
            break
        frames.append(frame)
        if frame["type"] == "cache_check":
            _send_response(resp_w, {"cache_hit": False, "provenance_hash": "abc"})
        elif frame["type"] == "persist":
            _send_response(resp_w, {"ok": True, "uri": "strata://test/c1"})
        elif frame["type"] == "batch_end":
            break

    thread.join(timeout=5)
    assert not thread.is_alive(), "harness thread did not exit"
    assert not errors, f"harness raised: {errors!r}"

    cell_errors = [f for f in frames if f["type"] == "cell_error"]
    assert not cell_errors, f"strata not injected in batch path: {cell_errors}"

    persist = next(f["payload"] for f in frames if f["type"] == "persist")
    assert "client_type" in persist["outputs"]
    assert persist["outputs"]["client_type"]["preview"] == "StrataClient"
    # The injected client is an input, not a cell output.
    assert "strata" not in persist["outputs"]
