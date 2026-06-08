"""Unit tests for ``pool_worker.execute_harness`` — the in-process cell
executor that the warm pool dispatches into.

These tests exercise the manifest → result function directly without
spinning up the warm pool itself; the pool's job is just to feed a
manifest path through stdin and read the result line, so the contract
worth pinning here is the result dict produced for tricky input states.
"""

from __future__ import annotations

from pathlib import Path

from strata.notebook.pool_worker import execute_harness


class TestExecuteHarnessRdsInput:
    """An R-only RDS upstream consumed by a Python cell must surface
    the structured ``StrataRArtifactError`` instead of swallowing it
    and regressing to ``NameError: name 'fit' is not defined`` once
    the cell body runs.

    The warm pool is the default WebSocket execution path, so a
    swallow here breaks the common user experience even when the
    cold-subprocess harness (``harness.py``) handles it correctly.
    """

    def test_rds_input_surfaces_structured_error(self, tmp_path: Path) -> None:
        rds_path = tmp_path / "fit.rds"
        # Dispatch rejects on content_type, not file shape — these bytes
        # never get parsed.
        rds_path.write_bytes(b"\x1f\x8b\x08\x00fakerds")

        manifest = {
            "source": "result = fit + 1",  # would NameError if swallowed
            "inputs": {
                "fit": {
                    "content_type": "application/x-r-rds",
                    "file": "fit.rds",
                }
            },
            "output_dir": str(tmp_path),
        }

        result = execute_harness(manifest)

        assert result["success"] is False
        # The structured message — variable name + saveRDS + data.frame
        # suggestion — gives the user the actionable fix instead of a
        # bare NameError.
        error = result["error"]
        assert "fit" in error
        assert "saveRDS" in error
        assert "data.frame" in error
        # Critical regression assertion: the previous behaviour swallowed
        # the deserialize error into stderr and the cell body then raised
        # NameError. The fix must surface the structured error type
        # instead.
        assert "NameError" not in error
        assert "StrataRArtifactError" in error


class TestExecuteHarnessTableInjection:
    """``@table`` declarations must inject ``<name>`` and
    ``<name>_snapshot`` into the warm-worker namespace.

    The warm pool is the default WebSocket execution path. It injected
    mounts but not tables, so an ``@table`` cell run through the pool
    failed with ``NameError`` for the injected URI variable while the
    cold ``harness.py`` path (which injects both) worked — the cell body
    references ``trips`` before it is ever defined.
    """

    def test_table_vars_injected(self, tmp_path: Path) -> None:
        manifest = {
            # Would NameError on either variable if injection is skipped.
            "source": "uri = trips\nsnap = trips_snapshot",
            "inputs": {},
            "output_dir": str(tmp_path),
            "tables": {
                "trips": {
                    "uri": "file:///wh#nyc.trips",
                    "snapshot_id": 2558063584752979421,
                }
            },
        }

        result = execute_harness(manifest)

        assert result["success"] is True, result.get("error")
        assert "uri" in result["variables"]
        assert "snap" in result["variables"]
