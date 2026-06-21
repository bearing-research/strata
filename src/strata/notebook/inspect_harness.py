"""Harness script that runs inside the inspect REPL subprocess.

Receives a manifest JSON file (path as ``argv[1]``), pre-loads the cell's input
variables into a namespace, then reads JSON command lines from stdin, evaluates
them, and writes JSON results to stdout. The process stays alive until an
explicit ``close`` command, so successive evaluations reuse the loaded inputs.

It runs in the notebook's venv, so it cannot ``import strata`` — instead it
loads ``serializer.py`` from the same directory via ``importlib.util`` (the
package directory, which is this file's parent).

Communication protocol
-----------------------
Parent -> Child: JSON lines on stdin  ``{"expr": "df.describe()"}``
Child -> Parent: JSON lines on stdout ``{"ok": true, "result": "...", "type": "str"}``
                                  or  ``{"ok": false, "error": "..."}``

Special commands::

    {"cmd": "ping"}  -> {"ok": true, "result": "pong"}
    {"cmd": "close"} -> process exits
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import traceback
from pathlib import Path


def _load_serializer():
    """Load the sibling ``serializer.py`` by absolute file path."""
    path = Path(__file__).parent / "serializer.py"
    spec = importlib.util.spec_from_file_location("_nb_serializer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ser = _load_serializer()


def _repr_value(value, max_len=4000):
    """Produce a display-friendly representation of a value."""
    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            return value.to_string(max_rows=20, max_cols=15)
        if isinstance(value, pd.Series):
            return value.to_string(max_rows=20)
    except ImportError:
        pass

    try:
        import pyarrow as pa

        if isinstance(value, pa.Table):
            return value.to_pandas().to_string(max_rows=20, max_cols=15)
    except ImportError:
        pass

    r = repr(value)
    if len(r) > max_len:
        r = r[:max_len] + "... (truncated)"
    return r


def _detect_type(value):
    """Return a human-readable type string for a value."""
    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            return f"DataFrame ({value.shape[0]} rows x {value.shape[1]} cols)"
        if isinstance(value, pd.Series):
            return f"Series ({len(value)} items)"
    except ImportError:
        pass
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return f"ndarray {value.shape}"
    except ImportError:
        pass
    return type(value).__name__


def _load_namespace(manifest):
    """Build the evaluation namespace from the manifest's input blobs."""
    namespace = {}
    inputs = manifest.get("inputs", {})
    output_dir = manifest.get("output_dir", "/tmp")

    for var_name, spec in inputs.items():
        content_type = spec.get("content_type", "")
        file_name = spec.get("file", "")
        if not file_name:
            continue
        full_path = Path(output_dir) / file_name
        if not full_path.exists():
            continue
        try:
            namespace[var_name] = _ser.deserialize_value(content_type, full_path)
        except Exception as e:
            namespace[var_name] = f"<load error: {e}>"
    return namespace


def _emit(response):
    """Write one JSON response line to stdout and flush."""
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def _evaluate(expr, namespace):
    """Evaluate an expression (or statement) and return a response dict.

    Tries ``eval`` first; on ``SyntaxError`` falls back to ``exec`` so plain
    statements work too. stdout produced during evaluation is captured and
    returned under ``stdout`` (expressions) or as the result (statements).
    """
    old_stdout, old_stderr = sys.stdout, sys.stderr
    capture_out = io.StringIO()
    try:
        sys.stdout = capture_out
        sys.stderr = io.StringIO()
        try:
            result = eval(expr, namespace)  # noqa: S307 — user-driven inspect REPL
        except SyntaxError:
            # Not an expression — run it as a statement.
            exec(expr, namespace)  # noqa: S102 — user-driven inspect REPL
            sys.stdout, sys.stderr = old_stdout, old_stderr
            stdout_text = capture_out.getvalue()
            return {
                "ok": True,
                "result": stdout_text or "(no output)",
                "type": "None",
            }
        sys.stdout, sys.stderr = old_stdout, old_stderr
        response = {
            "ok": True,
            "result": _repr_value(result),
            "type": _detect_type(result),
        }
        stdout_text = capture_out.getvalue()
        if stdout_text:
            response["stdout"] = stdout_text
        return response
    except Exception:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        return {"ok": False, "error": traceback.format_exc()}


def main():
    """Read JSON commands from stdin, evaluate them, write results to stdout."""
    with open(sys.argv[1]) as f:
        manifest = json.load(f)

    namespace = _load_namespace(manifest)

    # Signal ready.
    _emit({"ok": True, "result": "ready", "type": "str"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as e:
            _emit({"ok": False, "error": f"JSON parse error: {e}"})
            continue

        if cmd.get("cmd") == "ping":
            _emit({"ok": True, "result": "pong", "type": "str"})
            continue
        if cmd.get("cmd") == "close":
            break

        expr = cmd.get("expr", "")
        if not expr:
            _emit({"ok": False, "error": "Empty expression"})
            continue

        _emit(_evaluate(expr, namespace))


if __name__ == "__main__":
    main()
