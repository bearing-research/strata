"""Importing the CLI leaves the server's notebook stack unloaded."""

from __future__ import annotations

import json
import subprocess
import sys


def test_the_cli_does_not_import_the_notebook_server():
    # Each CLI call pays its imports; routes alone pulled in FastAPI and the executor.
    probe = (
        "import json, sys, strata.cli;"
        "print(json.dumps([m for m in ('strata.notebook.routes', 'fastapi') if m in sys.modules]))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout
    assert json.loads(out) == []
