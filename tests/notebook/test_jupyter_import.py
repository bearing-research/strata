"""Tests for ``strata.notebook.jupyter_import``.

PR 1 coverage: parse + convert markdown / code cells, ``;``-suppression
handling, source-order preservation. Magic translation, shell-command
extraction, and dep capture get their own test file in PR 2.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from strata.notebook.jupyter_import import (
    _SUPPRESSED_COMMENT,
    import_notebook,
)
from strata.notebook.parser import parse_notebook


def _make_ipynb(
    tmp_path: Path,
    cells: list[dict],
    name: str = "input.ipynb",
) -> Path:
    """Write a minimal nbformat-4 notebook with the supplied cells."""
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = tmp_path / name
    path.write_text(json.dumps(nb), encoding="utf-8")
    return path


def _code_cell(source: str | list[str]) -> dict:
    return {
        "cell_type": "code",
        "source": source,
        "outputs": [],
        "execution_count": None,
        "metadata": {},
    }


def _md_cell(source: str | list[str]) -> dict:
    return {"cell_type": "markdown", "source": source, "metadata": {}}


def test_import_creates_notebook_dir_with_pyproject_and_toml(tmp_path: Path) -> None:
    ipynb = _make_ipynb(tmp_path, [_code_cell("x = 1\n")])

    result = import_notebook(ipynb)

    assert result.notebook_dir.is_dir()
    assert (result.notebook_dir / "notebook.toml").is_file()
    assert (result.notebook_dir / "pyproject.toml").is_file()
    assert (result.notebook_dir / "cells").is_dir()


def test_import_converts_markdown_and_code_cells_in_source_order(tmp_path: Path) -> None:
    ipynb = _make_ipynb(
        tmp_path,
        [
            _md_cell("# Intro\n\nFirst cell.\n"),
            _code_cell("import numpy as np\n"),
            _md_cell("## Next step\n"),
            _code_cell("x = np.arange(5)\n"),
        ],
    )

    result = import_notebook(ipynb)

    assert result.markdown_cells == 2
    assert result.code_cells == 2
    assert result.skipped_cells == []

    nb = parse_notebook(result.notebook_dir)
    assert len(nb.cells) == 4
    languages = [c.language for c in nb.cells]
    assert languages == ["markdown", "python", "markdown", "python"]
    # Source survives intact through the round-trip.
    assert "# Intro" in nb.cells[0].source
    assert "import numpy as np" in nb.cells[1].source
    assert "## Next step" in nb.cells[2].source
    assert "x = np.arange(5)" in nb.cells[3].source


def test_import_preserves_trailing_semicolon_display_suppression(tmp_path: Path) -> None:
    """Jupyter convention: ``df;`` evaluates ``df`` but suppresses the
    auto-displayed value. Strata's harness auto-displays any final
    bare expression, so the converter has to rewrite the cell so the
    last statement isn't an ``ast.Expr``."""
    ipynb = _make_ipynb(
        tmp_path,
        [_code_cell("import pandas as pd\ndf = pd.DataFrame()\ndf;")],
    )

    result = import_notebook(ipynb)

    assert result.suppressed_outputs == 1
    nb = parse_notebook(result.notebook_dir)
    source = nb.cells[0].source
    # Trailing ; gone, replaced with an explicit pass so harness skips display.
    assert source.rstrip().endswith("pass")
    assert "df;" not in source
    assert _SUPPRESSED_COMMENT in source


def test_import_detects_suppression_followed_by_inline_comment(tmp_path: Path) -> None:
    """Real notebooks routinely write ``df;  # don't print`` — the ``;``
    is still suppression, the comment is just commentary."""
    ipynb = _make_ipynb(
        tmp_path,
        [_code_cell("import pandas as pd\ndf = pd.DataFrame()\ndf;  # quiet\n")],
    )

    result = import_notebook(ipynb)
    assert result.suppressed_outputs == 1
    nb = parse_notebook(result.notebook_dir)
    assert nb.cells[0].source.rstrip().endswith("pass")


def test_import_passes_through_non_suppressed_last_expression(tmp_path: Path) -> None:
    """The harness handles auto-display natively; we should not wrap
    bare last expressions in ``display(...)``."""
    ipynb = _make_ipynb(tmp_path, [_code_cell("x = 1\nx + 1\n")])

    result = import_notebook(ipynb)
    assert result.suppressed_outputs == 0
    nb = parse_notebook(result.notebook_dir)
    assert "display(" not in nb.cells[0].source
    assert nb.cells[0].source.rstrip().endswith("x + 1")


def test_import_handles_source_as_list_of_lines(tmp_path: Path) -> None:
    """nbformat canonical shape is a list of lines (each ending with \\n
    except possibly the last). Hand-edited notebooks sometimes use a
    plain string. Both have to work."""
    ipynb = _make_ipynb(
        tmp_path,
        [_code_cell(["import numpy as np\n", "x = np.zeros(3)\n", "x"])],
    )

    result = import_notebook(ipynb)
    nb = parse_notebook(result.notebook_dir)
    src = nb.cells[0].source
    assert "import numpy as np" in src
    assert "x = np.zeros(3)" in src
    assert src.rstrip().endswith("x")


def test_import_skips_raw_cells_and_records_them(tmp_path: Path) -> None:
    ipynb = _make_ipynb(
        tmp_path,
        [
            _code_cell("x = 1\n"),
            {"cell_type": "raw", "source": "raw latex here", "metadata": {}},
            _code_cell("y = 2\n"),
        ],
    )

    result = import_notebook(ipynb)
    assert result.code_cells == 2
    assert result.skipped_cells == ["raw"]
    nb = parse_notebook(result.notebook_dir)
    # The raw cell is dropped; the two code cells survive in order.
    assert len(nb.cells) == 2


def test_import_returns_error_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        import_notebook(tmp_path / "does-not-exist.ipynb")


def test_import_writes_to_explicit_out_dir_when_provided(tmp_path: Path) -> None:
    ipynb = _make_ipynb(tmp_path, [_code_cell("x = 1\n")], name="source.ipynb")
    target = tmp_path / "my-imports" / "custom-name"

    result = import_notebook(ipynb, out_dir=target)
    assert result.notebook_dir.name == "custom-name"
    assert (result.notebook_dir / "notebook.toml").is_file()


def test_import_default_out_dir_uses_ipynb_stem(tmp_path: Path) -> None:
    ipynb = _make_ipynb(tmp_path, [_code_cell("x = 1\n")], name="my_analysis.ipynb")

    result = import_notebook(ipynb)
    # create_notebook slugifies (lower + spaces→underscores) so the
    # already-snake_case stem comes through unchanged.
    assert result.notebook_dir.name == "my_analysis"
    assert result.notebook_dir.parent == tmp_path


def test_strata_import_cli_writes_notebook_dir(tmp_path: Path) -> None:
    """End-to-end smoke through the ``strata import`` CLI subcommand."""
    ipynb = _make_ipynb(
        tmp_path,
        [_md_cell("# Title\n"), _code_cell("x = 1\nprint(x)\n")],
        name="cli_test.ipynb",
    )

    result = subprocess.run(
        [sys.executable, "-m", "strata.cli", "import", str(ipynb)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Imported" in result.stdout
    assert "1 code, 1 markdown" in result.stdout

    nb_dir = tmp_path / "cli_test"
    assert (nb_dir / "notebook.toml").is_file()
    nb = parse_notebook(nb_dir)
    assert len(nb.cells) == 2


def test_strata_import_cli_rejects_missing_file(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "strata.cli",
            "import",
            str(tmp_path / "missing.ipynb"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
