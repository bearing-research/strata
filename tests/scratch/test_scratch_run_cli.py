"""``strata scratch run`` / ``show``: the MCP tools for an agent that has only a shell."""

from __future__ import annotations

import json

from strata import scratch
from strata.cli import main


def test_scratch_run_starts_the_scratchpad_and_show_finds_what_it_holds(project, capsys):
    root = str(project)

    assert main(["scratch", "run", "-c", "x = 41\nx + 1", "--root", root]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "ok"
    assert first["value"] == 42
    assert "setup_ms" in first

    assert main(["scratch", "show", "x", "--root", root]) == 0
    held = json.loads(capsys.readouterr().out)
    assert "x = 41" in held["source"]

    assert main(["scratch", "run", "-c", "x + 1", "--root", root]) == 0
    again = json.loads(capsys.readouterr().out)
    assert again["value"] == 42
    assert "setup_ms" not in again
    assert scratch.find_daemon(project) is not None


def test_a_cell_that_fails_exits_1_with_its_error(project, capsys):
    assert main(["scratch", "run", "-c", "1 / 0", "--root", str(project)]) == 1
    failed = json.loads(capsys.readouterr().out)
    assert failed["status"] == "error"
    assert "ZeroDivisionError" in json.dumps(failed)


def test_a_source_file_and_a_cell_to_replace(project, tmp_path, capsys):
    root = str(project)
    source = tmp_path / "cell.py"
    source.write_text("y = 2\ny * 2")
    assert main(["scratch", "run", "--file", str(source), "--root", root]) == 0
    cell = json.loads(capsys.readouterr().out)["cell"]

    assert main(["scratch", "run", "-c", "y = 3\ny * 2", "--cell", cell, "--root", root]) == 0
    replaced = json.loads(capsys.readouterr().out)
    assert (replaced["cell"], replaced["value"]) == (cell, 6)
