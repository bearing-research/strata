"""Unit tests for ``pool_worker.execute_harness``, the warm pool's in-process executor.

The pool only pipes a manifest in and a result line out, so the result dict
is the contract pinned here.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from strata.notebook import pool_worker
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


def test_a_cells_sys_exit_fails_the_cell_and_the_worker_reports_it(tmp_path: Path) -> None:
    """The worker returns it as the cell's failure, and its streams are its own again."""
    import sys

    stdout, stderr = sys.stdout, sys.stderr
    result = execute_harness(
        {"source": "import sys\nprint('before')\nsys.exit(0)\n", "output_dir": str(tmp_path)}
    )

    assert result["success"] is False
    assert result["error"] == (
        "RuntimeError: the cell exited with status 0, which stops it before its end"
    )
    assert result["stdout"] == "before\n"
    assert (sys.stdout, sys.stderr) == (stdout, stderr)


def test_an_interrupt_still_returns_a_result(tmp_path: Path) -> None:
    import sys

    stdout = sys.stdout
    result = execute_harness({"source": "raise KeyboardInterrupt", "output_dir": str(tmp_path)})

    assert result["success"] is False
    assert result["error"] == "KeyboardInterrupt"
    assert sys.stdout is stdout


def _without_escape[T](call: Callable[[], T]) -> T:
    """An interrupt escaping a test would end the pytest session, not fail the test."""
    try:
        return call()
    except BaseException as exc:
        pytest.fail(f"escaped the worker: {exc!r}")


@pytest.mark.parametrize("raised", [KeyboardInterrupt, ValueError])
def test_a_report_that_fails_still_returns_the_print_trail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raised: type[BaseException]
) -> None:
    """As the cold harness writes one: with no result line the pool runs the cell again."""

    def broken(*args: object, **kwargs: object) -> dict:
        raise raised

    monkeypatch.setattr(pool_worker._traceback, "failure_fields", broken)
    manifest = {
        "source": "print('step 1 done')\nraise KeyError('x')\n",
        "output_dir": str(tmp_path),
    }
    result = _without_escape(lambda: execute_harness(manifest))

    assert result["success"] is False
    assert result["error"] == "the harness failed reporting the cell's failure"
    assert result["stdout"] == "step 1 done\n"


@pytest.mark.parametrize("closing", ["raise KeyboardInterrupt", "sys.exit(5)"])
def test_a_client_close_that_raises_keeps_the_cells_result(tmp_path: Path, closing: str) -> None:
    source = f"import sys\nx = 1\ndef close():\n    {closing}\nstrata.close = close\n"
    manifest = {"source": source, "output_dir": str(tmp_path), "strata_url": "http://127.0.0.1:9"}
    result = _without_escape(lambda: execute_harness(manifest))

    assert result["success"] is True, result
    assert "x" in result["variables"]


def test_main_prints_a_result_when_an_interrupt_escapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """On the protocol stream, though the cell's capture is still in place."""
    import io
    import sys

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")

    def interrupted(manifest: dict) -> dict:
        sys.stdout = io.StringIO()
        raise KeyboardInterrupt

    monkeypatch.setattr(pool_worker, "execute_harness", interrupted)
    monkeypatch.setattr(pool_worker._lifeline, "watch_lifeline", lambda: None)
    monkeypatch.setattr(pool_worker, "warm_imports", lambda imports: None)
    monkeypatch.setattr(sys, "argv", ["pool_worker.py", str(tmp_path)])
    monkeypatch.setattr(sys, "stdin", io.StringIO(f"{manifest}\n"))
    monkeypatch.setattr(sys, "stdout", sys.stdout)

    _without_escape(pool_worker.main)

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "ready"
    result = json.loads(lines[1])
    assert result["success"] is False
    assert result["error"] == "Pool worker error: KeyboardInterrupt"


class _TornStdout:
    """Writes half of the result line, then is interrupted: Ctrl-C mid-write."""

    def __init__(self) -> None:
        self.text = ""
        self.torn = False

    def write(self, text: str) -> int:
        if not self.torn and text.startswith("{"):
            self.torn = True
            self.text += text[: len(text) // 2]
            raise KeyboardInterrupt
        self.text += text
        return len(text)

    def flush(self) -> None:
        return None


def _run_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execute: Callable[[dict], dict], out: object
) -> None:
    import io
    import sys

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(pool_worker, "execute_harness", execute)
    monkeypatch.setattr(pool_worker._lifeline, "watch_lifeline", lambda: None)
    monkeypatch.setattr(pool_worker, "warm_imports", lambda imports: None)
    monkeypatch.setattr(sys, "argv", ["pool_worker.py", str(tmp_path)])
    monkeypatch.setattr(sys, "stdin", io.StringIO(f"{manifest}\n"))
    monkeypatch.setattr(sys, "stdout", out)
    _without_escape(pool_worker.main)


def test_a_torn_result_line_is_followed_by_a_whole_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = _TornStdout()
    _run_main(tmp_path, monkeypatch, lambda manifest: {"success": True, "variables": {}}, out)

    ready, torn, line = out.text.split("\n")[:3]
    assert ready == "ready"
    with pytest.raises(ValueError):
        json.loads(torn)
    assert json.loads(line)["error"] == "Pool worker error: KeyboardInterrupt"


def test_an_interrupt_while_the_error_line_is_made_still_writes_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io

    def interrupted(manifest: dict) -> dict:
        raise KeyboardInterrupt

    def again(exc: BaseException) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(pool_worker._traceback, "error_text", again)
    out = io.StringIO()
    _run_main(tmp_path, monkeypatch, interrupted, out)

    result = json.loads(out.getvalue().splitlines()[1])
    assert result["success"] is False
    assert result["error"] == "Pool worker error: KeyboardInterrupt"
