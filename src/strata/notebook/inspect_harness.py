"""Harness script that runs inside the inspect REPL subprocess.

Loads the cell's inputs from the manifest at ``argv[1]``, then evaluates JSON
command lines from stdin until ``close``. Runs in the notebook's venv, so it
loads the sibling ``serializer.py`` by path instead of importing ``strata``.

Protocol (JSON lines)::

    stdin:  {"expr": "df.describe()"}
    stdout: {"ok": true, "result": "...", "type": "str"}  or  {"ok": false, "error": "..."}
    {"cmd": "ping"}  -> {"ok": true, "result": "pong"}
    {"cmd": "close"} -> process exits
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
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


def _enter_work_dir(manifest):
    """Evaluate where the cell ran: its notebook's ``work/`` (``work_dir.enter``)."""
    work_dir = manifest.get("work_dir")
    if work_dir:
        os.chdir(work_dir)


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


# Replies go to the original stdout through a private fd; ``main`` points fd 1 at
# devnull so output written straight to it (``os.system``, C code) cannot pose as one.
_replies = sys.stdout


def _take_stdout():
    """Keep the parent's stdout pipe for replies only; return it as a text stream."""
    reply_fd = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    return os.fdopen(reply_fd, "w")


def _emit(response):
    """Write one JSON response line to the reply stream and flush."""
    _replies.write(json.dumps(response) + "\n")
    _replies.flush()


def _evaluate(expr, namespace):
    """Evaluate an expression (or statement) and return a response dict.

    Falls back from ``eval`` to ``exec`` on ``SyntaxError``. Captured stdout is
    returned under ``stdout`` for expressions, or as the result for statements.
    """
    old_stdout, old_stderr = sys.stdout, sys.stderr
    capture_out = io.StringIO()
    try:
        sys.stdout = capture_out
        sys.stderr = io.StringIO()
        try:
            result = eval(expr, namespace)  # noqa: S307 (user-driven inspect REPL)
        except SyntaxError:
            # Not an expression: run it as a statement.
            exec(expr, namespace)  # noqa: S102 (user-driven inspect REPL)
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
    global _replies
    _replies = _take_stdout()
    with open(sys.argv[1]) as f:
        manifest = json.load(f)

    _enter_work_dir(manifest)
    namespace = _load_namespace(manifest)

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
