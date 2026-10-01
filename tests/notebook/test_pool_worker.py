"""Unit tests for ``pool_worker.execute_harness``, the warm pool's in-process executor.

The pool only pipes a manifest in and a result line out, so the result dict
is the contract pinned here.
"""

from __future__ import annotations

from pathlib import Path

from strata.notebook.pool_worker import execute_harness


class TestExecuteHarnessRdsInput:
    """An RDS upstream raises ``StrataRArtifactError``, not a later ``NameError``.

    The warm pool is the default WebSocket path, so a swallow here breaks the
    common case even when ``harness.py`` is correct.
    """

    def test_rds_input_surfaces_structured_error(self, tmp_path: Path) -> None:
        rds_path = tmp_path / "fit.rds"
        # Dispatch rejects on content_type, not file shape; these bytes are never parsed.
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
        # The structured message (variable name, saveRDS, data.frame suggestion)
        # gives the user the actionable fix instead of a bare NameError.
        error = result["error"]
        assert "fit" in error
        assert "saveRDS" in error
        assert "data.frame" in error
        # The deserialize error must surface as the structured error, not be
        # swallowed into stderr so the cell body raises NameError.
        assert "NameError" not in error
        assert "StrataRArtifactError" in error


class TestExecuteHarnessTableInjection:
    """``@table`` injects ``<name>`` and ``<name>_snapshot`` into the warm namespace,
    matching ``harness.py``.
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


class TestExecuteHarnessClientInjection:
    """``strata_url`` injects an ambient ``strata`` client, excluded from outputs.

    It is closed after the cell: the warm process is reused, so a leaked
    ``httpx.Client`` would accumulate sockets.
    """

    def test_client_injected_and_not_an_output(self, tmp_path: Path) -> None:
        manifest = {
            # NameError if strata is not injected; the derived var proves it.
            "source": "client_type = type(strata).__name__",
            "inputs": {},
            "output_dir": str(tmp_path),
            "strata_url": "http://127.0.0.1:8765",
        }

        result = execute_harness(manifest)

        assert result["success"] is True, result.get("error")
        assert "client_type" in result["variables"]
        # The injected client is an input, not a cell output.
        assert "strata" not in result["variables"]

    def test_absent_without_strata_url(self, tmp_path: Path) -> None:
        # No strata_url → no injection → referencing strata NameErrors.
        manifest = {
            "source": "x = strata",
            "inputs": {},
            "output_dir": str(tmp_path),
        }

        result = execute_harness(manifest)

        assert result["success"] is False
        assert "NameError" in result["error"]


class TestExecuteHarnessClientCellId:
    """The injected client carries the cell id so named writes can stamp ``nb_cell``."""

    def test_cell_id_reaches_injected_client(self, tmp_path: Path) -> None:
        manifest = {
            "source": "cid = strata._cell_id",
            "inputs": {},
            "output_dir": str(tmp_path),
            "strata_url": "http://127.0.0.1:8765",
            "strata_cell_id": "cell-xyz",
        }
        result = execute_harness(manifest)
        assert result["success"] is True, result.get("error")
        assert result["variables"]["cid"]["preview"] == "cell-xyz"

    def test_remote_store_headers_reach_injected_client(self, tmp_path: Path) -> None:
        """strata_headers (remote store auth/tenant) reach the warm worker's client."""
        manifest = {
            "source": "principal = strata._headers.get('X-Strata-Principal', '')",
            "inputs": {},
            "output_dir": str(tmp_path),
            "strata_url": "http://remote-store:8765",
            "strata_headers": {"X-Strata-Principal": "alice", "X-Tenant-ID": "team-a"},
        }
        result = execute_harness(manifest)
        assert result["success"] is True, result.get("error")
        assert result["variables"]["principal"]["preview"] == "alice"


class TestExecuteHarnessDisplayDeduplication:
    """A cell ending in a bare consumed variable must not serialize it twice, as in the cold
    path."""

    def test_display_that_is_a_variable_reuses_its_payload(self, tmp_path: Path) -> None:
        manifest = {
            "source": "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})\ndf",
            "inputs": {},
            "output_dir": str(tmp_path),
        }

        result = execute_harness(manifest)

        assert result["success"] is True
        variable = result["variables"]["df"]
        display = result["displays"][0]
        assert display["file"] == variable["file"]
        assert not list(tmp_path.glob("__display__*"))

    def test_display_that_is_not_a_variable_is_written(self, tmp_path: Path) -> None:
        manifest = {
            "source": ("import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})\ndf.head(2)"),
            "inputs": {},
            "output_dir": str(tmp_path),
        }

        result = execute_harness(manifest)

        assert result["success"] is True
        assert result["displays"][0]["file"] != result["variables"]["df"]["file"]
        assert list(tmp_path.glob("__display__*"))
