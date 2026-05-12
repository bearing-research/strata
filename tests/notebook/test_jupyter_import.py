"""Tests for ``strata.notebook.jupyter_import``.

PR 1 coverage: parse + convert markdown / code cells, ``;``-suppression
handling, source-order preservation.

PR 2 coverage: line-magic and cell-magic translation per the table,
``!shell`` handling, dependency capture from sibling
``requirements.txt`` / ``pyproject.toml`` and from ``pip install``
lines extracted from cells.
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


# ---------------------------------------------------------------------------
# Magic translation (PR 2)


def test_drops_matplotlib_inline_magic(tmp_path: Path) -> None:
    """%matplotlib inline is decorative in Strata — figures are
    captured via the display protocol regardless."""
    ipynb = _make_ipynb(
        tmp_path,
        [_code_cell("%matplotlib inline\nimport matplotlib.pyplot as plt\n")],
    )
    result = import_notebook(ipynb)
    assert "%matplotlib" in " ".join(result.translated_magics)
    nb = parse_notebook(result.notebook_dir)
    src = nb.cells[0].source
    assert "%matplotlib" not in src
    assert "import matplotlib.pyplot as plt" in src


def test_drops_load_ext_and_autoreload(tmp_path: Path) -> None:
    ipynb = _make_ipynb(
        tmp_path,
        [_code_cell("%load_ext autoreload\n%autoreload 2\nx = 1\n")],
    )
    result = import_notebook(ipynb)
    nb = parse_notebook(result.notebook_dir)
    src = nb.cells[0].source
    assert "%load_ext" not in src
    assert "%autoreload" not in src
    assert "x = 1" in src
    assert len(result.translated_magics) >= 2


def test_translates_env_magic_to_cell_annotation(tmp_path: Path) -> None:
    ipynb = _make_ipynb(tmp_path, [_code_cell("%env DEBUG=1\nx = 1\n")])
    result = import_notebook(ipynb)
    nb = parse_notebook(result.notebook_dir)
    src = nb.cells[0].source
    assert "# @env DEBUG=1" in src
    assert "%env" not in src


def test_strips_line_timeit_keeps_body(tmp_path: Path) -> None:
    ipynb = _make_ipynb(tmp_path, [_code_cell("%timeit sum(range(100))\n")])
    result = import_notebook(ipynb)
    nb = parse_notebook(result.notebook_dir)
    src = nb.cells[0].source
    assert "sum(range(100))" in src
    assert "%timeit" not in src


def test_translates_cell_magic_bash_to_subprocess(tmp_path: Path) -> None:
    ipynb = _make_ipynb(
        tmp_path,
        [_code_cell("%%bash\necho hello\nls -la\n")],
    )
    result = import_notebook(ipynb)
    nb = parse_notebook(result.notebook_dir)
    src = nb.cells[0].source
    assert "subprocess" in src
    assert "shell=True" in src
    assert "echo hello" in src
    assert "%%bash" not in src


def test_translates_cell_magic_writefile(tmp_path: Path) -> None:
    ipynb = _make_ipynb(
        tmp_path,
        [_code_cell("%%writefile output.txt\nline 1\nline 2\n")],
    )
    result = import_notebook(ipynb)
    nb = parse_notebook(result.notebook_dir)
    src = nb.cells[0].source
    assert "write_text" in src
    assert "output.txt" in src
    assert "line 1" in src


def test_cell_magic_timeit_recurses_on_body(tmp_path: Path) -> None:
    """%%timeit\\n<body> → just <body>. The body itself goes through
    the regular line-magic pass so a `%env` inside it would also
    translate."""
    ipynb = _make_ipynb(
        tmp_path,
        [_code_cell("%%timeit -n 1\nx = sum(range(100))\nx\n")],
    )
    result = import_notebook(ipynb)
    nb = parse_notebook(result.notebook_dir)
    src = nb.cells[0].source
    assert "%%timeit" not in src
    assert "x = sum(range(100))" in src
    # bare `x` still pass-throughs (not display-wrapped)
    assert src.rstrip().endswith("x")


def test_drops_javascript_and_html_cell_magics(tmp_path: Path) -> None:
    ipynb = _make_ipynb(
        tmp_path,
        [
            _code_cell("%%javascript\nalert('hi')\n"),
            _code_cell("%%html\n<b>bold</b>\n"),
        ],
    )
    result = import_notebook(ipynb)
    assert len(result.dropped_magics) == 2
    nb = parse_notebook(result.notebook_dir)
    assert "alert" not in nb.cells[0].source
    assert "<b>bold</b>" not in nb.cells[1].source


def test_unsupported_line_magic_dropped_with_comment(tmp_path: Path) -> None:
    ipynb = _make_ipynb(tmp_path, [_code_cell("%who\nx = 1\n")])
    result = import_notebook(ipynb)
    assert "%who" in " ".join(result.dropped_magics)
    nb = parse_notebook(result.notebook_dir)
    src = nb.cells[0].source
    assert "# strata: unsupported magic '%who' dropped" in src
    assert "%who" not in src.replace("'%who'", "")
    assert "x = 1" in src


# ---------------------------------------------------------------------------
# Shell commands + pip-install dep capture


def test_pip_install_magic_captures_packages(tmp_path: Path) -> None:
    ipynb = _make_ipynb(
        tmp_path,
        [_code_cell("%pip install requests pandas==2.0\nimport requests\n")],
    )
    result = import_notebook(ipynb)
    assert "requests" in result.captured_deps
    assert "pandas==2.0" in result.captured_deps
    nb = parse_notebook(result.notebook_dir)
    src = nb.cells[0].source
    assert "%pip install" not in src
    assert "import requests" in src


def test_pip_install_shell_form_captures_packages(tmp_path: Path) -> None:
    ipynb = _make_ipynb(
        tmp_path,
        [_code_cell("!pip install httpx\nimport httpx\n")],
    )
    result = import_notebook(ipynb)
    assert "httpx" in result.captured_deps
    nb = parse_notebook(result.notebook_dir)
    src = nb.cells[0].source
    assert "!pip install" not in src
    assert "import httpx" in src


def test_pip_install_does_not_capture_subcommand_word(tmp_path: Path) -> None:
    """Regression: ``%pip install httpx`` must capture ``httpx``, NOT
    the literal subcommand word ``install``. Caught by end-to-end run
    on a fixture with a single ``%pip install`` line."""
    ipynb = _make_ipynb(tmp_path, [_code_cell("%pip install httpx\n")])
    result = import_notebook(ipynb)
    assert result.captured_deps == ["httpx"]
    assert "install" not in result.captured_deps


def test_pip_list_and_uninstall_are_dropped(tmp_path: Path) -> None:
    """Only ``%pip install`` captures deps; other subcommands surface
    as dropped magics rather than getting their args treated as packages."""
    ipynb = _make_ipynb(
        tmp_path,
        [_code_cell("%pip list\n%pip uninstall pandas -y\n")],
    )
    result = import_notebook(ipynb)
    assert result.captured_deps == []
    assert any("list" in m for m in result.dropped_magics)
    assert any("uninstall" in m for m in result.dropped_magics)


def test_pip_install_strips_flags(tmp_path: Path) -> None:
    ipynb = _make_ipynb(
        tmp_path,
        [_code_cell("!pip install -U --quiet requests scikit-learn>=1.0\n")],
    )
    result = import_notebook(ipynb)
    assert result.captured_deps == ["requests", "scikit-learn>=1.0"]


def test_pip_install_skips_flag_value_pairs(tmp_path: Path) -> None:
    """``-r req.txt`` consumes its following arg, so the next token
    is NOT a package name."""
    ipynb = _make_ipynb(
        tmp_path,
        [_code_cell("!pip install -r requirements-extra.txt requests\n")],
    )
    result = import_notebook(ipynb)
    # ``requirements-extra.txt`` got consumed by -r, ``requests`` stays
    assert result.captured_deps == ["requests"]


def test_other_shell_commands_dropped_with_comment(tmp_path: Path) -> None:
    ipynb = _make_ipynb(
        tmp_path,
        [_code_cell("!ls /data\n!apt-get install foo\nimport sys\n")],
    )
    result = import_notebook(ipynb)
    assert len(result.dropped_shells) == 2
    nb = parse_notebook(result.notebook_dir)
    src = nb.cells[0].source
    assert "# strata: shell command dropped: !ls" in src
    assert "import sys" in src


# ---------------------------------------------------------------------------
# Sibling-file dep capture


def test_captures_deps_from_sibling_requirements_txt(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text(
        "# pinned for repro\nrequests==2.31.0\npandas\n-e ./local\n",
        encoding="utf-8",
    )
    ipynb = _make_ipynb(tmp_path, [_code_cell("x = 1\n")])

    result = import_notebook(ipynb)
    assert "requests==2.31.0" in result.captured_deps
    assert "pandas" in result.captured_deps
    # -e ./local is editable install, skipped
    assert "-e ./local" not in result.captured_deps

    # Deps land in the new notebook's pyproject.toml so first `uv sync`
    # picks them up. We don't run `uv add` ourselves.
    pyproject = (result.notebook_dir / "pyproject.toml").read_text()
    assert "requests==2.31.0" in pyproject
    assert "pandas" in pyproject


def test_captures_deps_from_sibling_pyproject_toml(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0"\n'
        'dependencies = ["scikit-learn", "matplotlib>=3.8"]\n',
        encoding="utf-8",
    )
    ipynb = _make_ipynb(tmp_path, [_code_cell("x = 1\n")])

    result = import_notebook(ipynb)
    assert "scikit-learn" in result.captured_deps
    assert "matplotlib>=3.8" in result.captured_deps


def test_deduplicates_deps_across_sources(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("requests\n", encoding="utf-8")
    ipynb = _make_ipynb(
        tmp_path,
        [_code_cell("!pip install requests pandas\n")],
    )
    result = import_notebook(ipynb)
    assert result.captured_deps.count("requests") == 1
    assert "pandas" in result.captured_deps


def test_suppression_still_works_after_magic_translation(tmp_path: Path) -> None:
    """A magic above a ``;``-suppressed expression shouldn't break
    the suppression rewrite."""
    ipynb = _make_ipynb(
        tmp_path,
        [_code_cell("%matplotlib inline\nimport pandas as pd\ndf = pd.DataFrame()\ndf;\n")],
    )
    result = import_notebook(ipynb)
    assert result.suppressed_outputs == 1
    nb = parse_notebook(result.notebook_dir)
    src = nb.cells[0].source
    assert "%matplotlib" not in src
    assert src.rstrip().endswith("pass")
