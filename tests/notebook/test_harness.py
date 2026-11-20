"""Tests for the harness subprocess execution."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_MINIMAL_PNG_LITERAL = (
    'b"\\x89PNG\\r\\n\\x1a\\n\\x00\\x00\\x00\\rIHDR\\x00\\x00\\x00\\x01\\x00\\x00\\x00\\x01'
    "\\x08\\x04\\x00\\x00\\x00\\xb5\\x1c\\x0c\\x02\\x00\\x00\\x00\\x0bIDATx\\xdac\\xfc\\xff"
    '\\x1f\\x00\\x03\\x03\\x02\\x00\\xef\\x9b\\xe0M\\x00\\x00\\x00\\x00IEND\\xaeB`\\x82"'
)
_MARKDOWN_LITERAL = '"# Title\\n\\n- one\\n- two"'


@pytest.fixture
def harness_script():
    """Get path to harness script."""
    return Path(__file__).parent.parent.parent / "src" / "strata" / "notebook" / "harness.py"


def run_harness(harness_path: Path, manifest: dict) -> dict:
    """Run the harness on ``manifest`` and return its result manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        manifest_path = tmpdir / "manifest.json"
        manifest["output_dir"] = str(tmpdir)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        subprocess.run(
            [sys.executable, str(harness_path), str(manifest_path)],
            cwd=str(tmpdir),
            capture_output=True,
        )

        result_path = tmpdir / "harness-result.json"
        with open(result_path) as f:
            return json.load(f)


class TestHarness:
    def test_harness_simple_assignment(self, harness_script):
        manifest = {"source": "x = 1 + 1", "inputs": {}}

        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        assert "x" in result["variables"]
        assert result["variables"]["x"]["content_type"] == "json/object"

    def test_harness_with_print(self, harness_script):
        manifest = {"source": 'print("Hello")\ny = 42', "inputs": {}}

        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        assert "Hello" in result["stdout"]
        assert "y" in result["variables"]

    def test_harness_with_error(self, harness_script):
        manifest = {"source": "z = 1 / 0", "inputs": {}}

        result = run_harness(harness_script, manifest)

        assert result["success"] is False
        assert "error" in result
        assert len(result["error"]) > 0

    def test_harness_keeps_the_print_trail_of_a_failing_cell(self, harness_script):
        """A cell that raised still reports its ``stdout``, the print trail that explains the
        failure, as the batch and pool paths do.
        """
        manifest = {
            "source": 'print("checkpoint A")\nprint("checkpoint B")\nraise KeyError("boom")',
            "inputs": {},
        }

        result = run_harness(harness_script, manifest)

        assert result["success"] is False
        assert "checkpoint A" in result["stdout"]
        assert "checkpoint B" in result["stdout"]

    @pytest.mark.parametrize(
        ("raised", "error"),
        [('KeyError("MAX")', "KeyError: 'MAX'"), ("ValueError()", "ValueError")],
    )
    def test_harness_names_the_exception_a_cell_raised(self, harness_script, raised, error):
        result = run_harness(harness_script, {"source": f"raise {raised}", "inputs": {}})

        assert result["error"] == error

    def test_harness_dataframe(self, harness_script):
        manifest = {
            "source": 'import pandas as pd\ndf = pd.DataFrame({"a": [1, 2, 3]})',
            "inputs": {},
        }

        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        assert "df" in result["variables"]
        assert result["variables"]["df"]["content_type"] == "arrow/ipc"
        assert result["variables"]["df"]["rows"] == 3

    def test_harness_bare_expression_no_display_mutation_warning(self, harness_script):
        """A bare trailing expression grows the injected ``display`` helper's buffer; that is not
        an in-place mutation of ``display``.
        """
        manifest = {"source": "x = 5\nx", "inputs": {}}

        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        warned = {w.get("var_name") for w in result.get("mutation_warnings", [])}
        assert "display" not in warned
        assert "Markdown" not in warned

    def test_harness_multiple_outputs(self, harness_script):
        manifest = {
            "source": """
x = 10
y = "hello"
z = [1, 2, 3]
""",
            "inputs": {},
        }

        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        var_names = set(result["variables"].keys())
        assert var_names == {"x", "y", "z"}

    def test_harness_dict_output(self, harness_script):
        manifest = {
            "source": 'data = {"count": 42, "name": "test"}',
            "inputs": {},
        }

        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        assert "data" in result["variables"]
        assert result["variables"]["data"]["content_type"] == "json/object"

    def test_harness_ignores_private(self, harness_script):
        manifest = {
            "source": """
public = 1
_private = 2
""",
            "inputs": {},
        }

        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        assert "public" in result["variables"]
        assert "_private" not in result["variables"]

    def test_harness_empty_output(self, harness_script):
        manifest = {"source": "# Just a comment", "inputs": {}}

        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        assert len(result["variables"]) == 0

    def test_harness_with_stderr(self, harness_script):
        manifest = {
            "source": """
import sys
print("error", file=sys.stderr)
x = 1
""",
            "inputs": {},
        }

        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        assert "error" in result["stderr"]

    def test_harness_reuses_the_variable_payload_for_an_identical_display(
        self, harness_script, tmp_path
    ):
        """A bare trailing variable must not be serialized a second time.

        The display and variable loops get the same object; a second write re-serializes it,
        and for a lazy or one-shot source gives different bytes. Runs the harness directly so the
        output directory survives.
        """
        manifest_path = tmp_path / "manifest.json"
        manifest = {
            "source": "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})\ndf",
            "inputs": {},
            "output_dir": str(tmp_path),
        }
        manifest_path.write_text(json.dumps(manifest))
        subprocess.run(
            [sys.executable, str(harness_script), str(manifest_path)],
            cwd=str(tmp_path),
            capture_output=True,
        )
        result = json.loads((tmp_path / "harness-result.json").read_text())

        assert result["success"] is True
        variable = result["variables"]["df"]
        display = result["displays"][0]
        assert display["file"] == variable["file"]
        assert not list(tmp_path.glob("__display__*"))

    def test_harness_still_writes_a_display_that_is_not_a_variable(self, harness_script, tmp_path):
        """The counterpart: an expression that is not a bound variable."""
        manifest_path = tmp_path / "manifest.json"
        manifest = {
            "source": ("import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})\ndf.head(2)"),
            "inputs": {},
            "output_dir": str(tmp_path),
        }
        manifest_path.write_text(json.dumps(manifest))
        subprocess.run(
            [sys.executable, str(harness_script), str(manifest_path)],
            cwd=str(tmp_path),
            capture_output=True,
        )
        result = json.loads((tmp_path / "harness-result.json").read_text())

        assert result["success"] is True
        assert result["displays"][0]["file"] != result["variables"]["df"]["file"]
        assert list(tmp_path.glob("__display__*"))

    def test_harness_captures_last_expression_png_display(self, harness_script):
        """Bare-expression values exposing _repr_png_ should be serialized as display output."""
        manifest = {
            "source": f"""
class Display:
    def _repr_png_(self):
        return {_MINIMAL_PNG_LITERAL}

Display()
""",
            "inputs": {},
        }

        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        assert "_" in result["variables"]
        assert result["variables"]["_"]["content_type"] == "image/png"
        assert result["variables"]["_"]["inline_data_url"].startswith("data:image/png;base64,")

    def test_harness_captures_last_expression_markdown_display(self, harness_script):
        """Bare-expression values exposing _repr_markdown_ should serialize as markdown display."""
        manifest = {
            "source": f"""
class Display:
    def _repr_markdown_(self):
        return {_MARKDOWN_LITERAL}

Display()
""",
            "inputs": {},
        }

        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        assert "_" in result["variables"]
        assert result["variables"]["_"]["content_type"] == "text/markdown"
        assert result["variables"]["_"]["markdown_text"] == "# Title\n\n- one\n- two"

    def test_harness_captures_display_call_png_output(self, harness_script):
        """Explicit display(...) side effects should populate the primary display output."""
        manifest = {
            "source": f"""
class Display:
    def _repr_png_(self):
        return {_MINIMAL_PNG_LITERAL}

display(Display())
""",
            "inputs": {},
        }

        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        assert "_" in result["variables"]
        assert result["variables"]["_"]["content_type"] == "image/png"

    def test_harness_captures_display_call_markdown_output(self, harness_script):
        """Injected display helpers should support Markdown(...) without imports."""
        manifest = {
            "source": """
display(Markdown("# Via helper\\n\\nRendered from display()."))
""",
            "inputs": {},
        }

        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        assert "_" in result["variables"]
        assert result["variables"]["_"]["content_type"] == "text/markdown"
        assert (
            result["variables"]["_"]["markdown_text"] == "# Via helper\n\nRendered from display()."
        )

    def test_harness_captures_pyplot_show_png_output(self, harness_script):
        """plt.show() should feed the primary display output when matplotlib is available."""
        pytest.importorskip("matplotlib.pyplot")
        manifest = {
            "source": """
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [1, 4, 9])
plt.show()
""",
            "inputs": {},
            "env": {"MPLBACKEND": "Agg"},
        }

        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        assert "_" in result["variables"]
        assert result["variables"]["_"]["content_type"] == "image/png"

    def test_harness_does_not_import_pyplot_for_a_cell_that_does_not(self, harness_script):
        """Importing it for every cell is slow, and on a fresh install its font-cache notice
        lands in the stderr of whichever cell ran first."""
        pytest.importorskip("matplotlib.pyplot")
        manifest = {
            "source": "import sys\nprint('matplotlib.pyplot' in sys.modules)\n",
            "inputs": {},
        }

        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        assert result["stdout"].strip() == "False"

    def test_harness_captures_multiple_visible_outputs_in_order(self, harness_script):
        """Visible outputs should be emitted in order with the last one preserved as '_'."""
        manifest = {
            "source": """
display(Markdown("# First"))
42
""",
            "inputs": {},
        }

        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        assert len(result["displays"]) == 2
        assert result["displays"][0]["content_type"] == "text/markdown"
        assert result["displays"][0]["markdown_text"] == "# First"
        assert result["displays"][1]["content_type"] == "json/object"
        assert result["displays"][1]["preview"] == 42
        assert result["variables"]["_"]["content_type"] == "json/object"
        assert result["variables"]["_"]["preview"] == 42

    def test_harness_complex_dataframe(self, harness_script):
        manifest = {
            "source": """import pandas as pd
df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'value': [1.5, 2.5, 3.5, 4.5, 5.5],
    'category': ['A', 'B', 'A', 'C', 'B']
})
""",
            "inputs": {},
        }

        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        assert result["variables"]["df"]["rows"] == 5
        assert set(result["variables"]["df"]["columns"]) == {"id", "value", "category"}
        assert "preview" in result["variables"]["df"]
        # Preview should have all 5 rows since it's small
        assert len(result["variables"]["df"]["preview"]) == 5

    def test_harness_function_definition(self, harness_script):
        manifest = {
            "source": """
def greet(name):
    return f"Hello, {name}!"

result = greet("World")
""",
            "inputs": {},
        }

        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        assert "greet" in result["variables"]
        assert "result" in result["variables"]
        assert result["variables"]["result"]["preview"] == "Hello, World!"

    def test_harness_with_imports(self, harness_script):
        manifest = {
            "source": """
import math
pi_value = math.pi
sqrt_2 = math.sqrt(2)
""",
            "inputs": {},
        }

        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        assert "pi_value" in result["variables"]
        assert "sqrt_2" in result["variables"]

    def test_harness_serialization_error_handled(self, harness_script):
        manifest = {
            "source": """
# Create a complex object that might not serialize well
import threading
lock = threading.Lock()
x = 1
""",
            "inputs": {},
        }

        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        # The lock may fail to serialize, but x must still be there.
        assert "x" in result["variables"]


class TestHarnessLoopUntil:
    """The harness evaluates ``@loop_until`` in the cell namespace after the body.

    Driven through the subprocess entry point, so the full manifest round trip is covered.
    """

    def test_loop_until_truthy_returns_until_reached(self, harness_script):
        manifest = {
            "source": "state = {'confidence': 0.95}",
            "inputs": {},
            "loop": {"until_expr": "state['confidence'] > 0.9"},
        }
        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        assert result["loop"]["until_reached"] is True
        assert result["loop"].get("error") is None

    def test_loop_until_falsy_reports_not_reached(self, harness_script):
        manifest = {
            "source": "state = {'confidence': 0.4}",
            "inputs": {},
            "loop": {"until_expr": "state['confidence'] > 0.9"},
        }
        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        assert result["loop"]["until_reached"] is False

    def test_loop_until_runtime_error_is_captured(self, harness_script):
        manifest = {
            "source": "x = 1",
            "inputs": {},
            "loop": {"until_expr": "undefined_name > 0"},
        }
        result = run_harness(harness_script, manifest)

        # A predicate failure is reported on the loop record instead of aborting
        # execution, so the executor decides how to surface it.
        assert result["success"] is True
        assert result["loop"]["until_reached"] is False
        assert "NameError" in result["loop"]["error"]

    def test_loop_absent_when_no_until_expr(self, harness_script):
        """A manifest without a ``loop`` key produces no ``loop`` record."""
        manifest = {"source": "x = 1", "inputs": {}}
        result = run_harness(harness_script, manifest)

        assert result["success"] is True
        assert "loop" not in result

    def test_loop_until_sees_carry_passed_as_input(self, harness_script, tmp_path):
        """The executor seeds the carry as a regular input (iteration k-1's artifact); the
        ``@loop_until`` expression sees the value after the body runs.
        """
        state_path = Path("state.pickle")
        seed = {"confidence": 0.5, "iteration": 0}

        # Write the seed as a pickle the harness can deserialize.
        import pickle

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            state_full = tmpdir / state_path.name
            with open(state_full, "wb") as f:
                pickle.dump(seed, f)

            manifest = {
                "source": "state = {**state, 'confidence': state['confidence'] + 0.5}",
                "inputs": {
                    "state": {
                        "content_type": "pickle/object",
                        "file": str(state_path),
                    }
                },
                "loop": {"until_expr": "state['confidence'] >= 1.0"},
                "output_dir": str(tmpdir),
            }
            manifest_path = tmpdir / "request_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

            subprocess.run(
                [sys.executable, str(harness_script), str(manifest_path)],
                cwd=str(tmpdir),
                capture_output=True,
            )

            with open(tmpdir / "harness-result.json") as f:
                result = json.load(f)

        assert result["success"] is True
        assert result["loop"]["until_reached"] is True


class TestHarnessRdsInput:
    """A Python cell consuming an R-only RDS input fails loudly.

    ``deserialize_inputs`` raises the structured ``StrataRArtifactError`` with its
    "re-export as data.frame" fix, rather than leaving a vague NameError for later.
    """

    def test_rds_input_surfaces_structured_error(self, harness_script):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # The bytes need not be a real RDS blob: the dispatcher rejects on
            # content_type, not file shape.
            rds_path = tmpdir / "fit.rds"
            rds_path.write_bytes(b"\x1f\x8b\x08\x00fakerds")

            manifest = {
                "source": "result = fit",  # would raise NameError if input swallowed
                "inputs": {
                    "fit": {
                        "content_type": "application/x-r-rds",
                        "file": "fit.rds",
                    }
                },
                "output_dir": str(tmpdir),
            }
            manifest_path = tmpdir / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

            subprocess.run(
                [sys.executable, str(harness_script), str(manifest_path)],
                cwd=str(tmpdir),
                capture_output=True,
            )

            with open(tmpdir / "harness-result.json") as f:
                result = json.load(f)

        assert result["success"] is False
        # The variable name and the saveRDS fix wording both come from
        # StrataRArtifactError; if the harness swallowed it, this would fail.
        assert "fit" in result["error"]
        assert "saveRDS" in result["error"]
        assert "data.frame" in result["error"]
        # The cell body never ran, so no NameError can leak.
        assert "NameError" not in result["error"]


class TestEnvAppliedBeforeDeserialization:
    """The notebook's ``[env]`` must be live before any input is decoded.

    Deserializing imports the producing library, and one that reads its config at import
    (jax and ``JAX_ENABLE_X64``) would take the server's environment and downcast float64
    inputs to float32.
    """

    def test_manifest_env_is_visible_while_inputs_are_deserialized(self, tmp_path, monkeypatch):
        import json as _json
        import os
        import sys as _sys

        from strata.notebook import harness

        seen: dict[str, str | None] = {}

        def _record(manifest):
            seen["value"] = os.environ.get("STRATA_ORDER_PROBE")
            return {}

        monkeypatch.setattr(harness, "deserialize_inputs", _record)

        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            _json.dumps(
                {
                    "source": "x = 1",
                    "inputs": {},
                    "env": {"STRATA_ORDER_PROBE": "1"},
                    "output_dir": str(tmp_path),
                }
            )
        )
        monkeypatch.setattr(_sys, "argv", ["harness.py", str(manifest_path)])
        monkeypatch.delenv("STRATA_ORDER_PROBE", raising=False)

        harness.main()

        assert seen["value"] == "1", (
            "inputs were deserialized before the notebook's [env] was applied"
        )
        assert os.environ.get("STRATA_ORDER_PROBE") is None, "the override must not leak"


@pytest.mark.parametrize("module", ["harness", "pool_worker"])
@pytest.mark.parametrize("before", [None, "outer"])
def test_the_cell_id_is_set_for_the_cell_and_put_back_after(module, before, monkeypatch):
    """A reused warm worker, or a test calling in-process, must not keep a cell's id."""
    import importlib

    exec_with_display = importlib.import_module(f"strata.notebook.{module}")._exec_with_display
    if before is None:
        monkeypatch.delenv("STRATA_CELL_ID", raising=False)
    else:
        monkeypatch.setenv("STRATA_CELL_ID", before)
    namespace: dict = {}

    seen = exec_with_display("import os\nos.environ['STRATA_CELL_ID']", namespace, "c9")
    assert seen == "c9"
    with pytest.raises(ZeroDivisionError):
        exec_with_display("1 / 0", namespace, "c9")
    # A run without an id does not pass on the one this process inherited.
    seen = exec_with_display("import os\nos.environ.get('STRATA_CELL_ID')", namespace, None)
    assert seen is None

    import os

    assert os.environ.get("STRATA_CELL_ID") == before


def test_around_body_wraps_the_cells_own_code_alone(monkeypatch):
    """The cold path's audit window: the body and ``loop_until``, as the warm worker's
    is, not the input snapshot or the mutation check after."""
    import contextlib

    from strata.notebook import harness

    log: list[tuple[str, bool]] = []
    inside: list[bool] = []

    @contextlib.contextmanager
    def around_body():
        log.append(("enter", bool(inside)))
        inside.append(True)
        try:
            yield
        finally:
            inside.pop()

    for name in ("snapshot_inputs", "detect_mutations"):
        real = getattr(harness._immut, name)

        def logged(*args, _name=name, _real=real, **kwargs):
            log.append((_name, bool(inside)))
            return _real(*args, **kwargs)

        monkeypatch.setattr(harness._immut, name, logged)
    real_exec = harness._exec_with_display

    def body(*args, **kwargs):
        log.append(("body", bool(inside)))
        return real_exec(*args, **kwargs)

    monkeypatch.setattr(harness, "_exec_with_display", body)
    monkeypatch.setattr(
        harness,
        "_eval_loop_until",
        lambda expr, namespace: log.append(("loop_until", bool(inside))) or {},
    )
    outputs, *_ = harness.execute_cell(
        "y = x + 1", {"x": 1}, loop_until_expr="y > 1", around_body=around_body
    )
    assert outputs["y"] == 2
    assert log == [
        ("snapshot_inputs", False),
        ("enter", False),
        ("body", True),
        ("loop_until", True),
        ("detect_mutations", False),
    ], log


_STOPS = "which stops it before its end"


@pytest.mark.parametrize(
    ("call", "message"),
    [
        ("sys.exit(0)", f"the cell exited with status 0, {_STOPS}"),
        ("sys.exit()", f"the cell exited, {_STOPS}"),
        ("sys.exit('')", f"the cell exited, {_STOPS}"),
        ("exit('bye')", f"the cell exited, {_STOPS}: bye"),
        # What Python prints for anything else: its str().
        ("sys.exit(ValueError('bad config'))", f"the cell exited, {_STOPS}: bad config"),
        (
            "sys.exit(type('Bad', (), {'__str__': lambda self: 1 / 0})())",
            f"the cell exited, {_STOPS}: <str() failed>",
        ),
        ("sys.exit(True)", f"the cell exited with status 1, {_STOPS}"),
        (
            "sys.exit(type('Bad', (), {'__str__': lambda self: sys.exit(2)})())",
            f"the cell exited, {_STOPS}: <str() failed>",
        ),
        # A message is the cell's own error text: whole, its lines kept.
        (
            "sys.exit('Config errors:\\n- missing key A\\n- missing key B')",
            f"the cell exited, {_STOPS}: Config errors:\n- missing key A\n- missing key B",
        ),
    ],
)
def test_a_cells_sys_exit_fails_the_cell(harness_script, call, message):
    """It stops the cell before its end; the harness still writes its result."""
    result = run_harness(harness_script, {"source": f"import sys\nx = 1\n{call}\n", "inputs": {}})

    assert result["success"] is False
    assert result["error"] == f"RuntimeError: {message}"
    assert result["variables"] == {}
    # From the cell's own frame, the harness's left out.
    traceback = result["traceback"]
    assert "line 3" in traceback, traceback
    assert "cell_traceback" not in traceback and "contextlib" not in traceback, traceback
    assert traceback.rstrip().endswith(f"RuntimeError: {message}"), traceback


@pytest.mark.parametrize(
    ("exits", "shown", "hidden"),
    [
        ("sys.exit(1)", "ValueError: invalid literal for int()", None),
        ("raise SystemExit(1) from None", None, "ValueError"),
        ("raise SystemExit(1) from KeyError('cfg')", "KeyError: 'cfg'", "ValueError"),
    ],
)
def test_a_cells_exit_shows_the_cause_python_would(harness_script, exits, shown, hidden):
    """What it was handling, unless ``from`` named another cause or ``from None`` hid it."""
    source = f"import sys\ntry:\n    int('x')\nexcept ValueError:\n    {exits}\n"
    traceback = run_harness(harness_script, {"source": source, "inputs": {}})["traceback"]

    if shown is not None:
        assert shown in traceback, traceback
    if hidden is not None:
        assert hidden not in traceback, traceback
    assert "cell_traceback" not in traceback, traceback


def test_a_sys_exit_in_loop_until_is_the_loops_error():
    from strata.notebook import harness

    namespace: dict = {}
    exec("import sys\ndef done():\n    sys.exit(0)\n", namespace)
    state = harness._eval_loop_until("done()", namespace)
    assert state["until_reached"] is False
    assert "the @loop_until predicate exited with status 0" in state["error"], state


def test_a_loop_until_error_whose_str_raises_is_still_reported():
    from strata.notebook import harness

    namespace: dict = {}
    exec(
        "class Unreadable(Exception):\n"
        "    def __str__(self):\n"
        "        raise ValueError\n"
        "def check():\n"
        "    raise Unreadable()\n",
        namespace,
    )
    state = harness._eval_loop_until("check()", namespace)
    assert state["error"] == (
        "@loop_until evaluation failed: Unreadable: <exception str() failed>"
    ), state


def test_a_status_too_large_to_print_is_still_a_status():
    import sys

    from strata.notebook import cell_traceback

    limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(4300)
    try:
        text = cell_traceback._exited("the cell", 10**5000)
    finally:
        sys.set_int_max_str_digits(limit)
    assert text == "the cell exited with status <too large to print>, which stops it before its end"


def test_a_long_exit_text_is_capped():
    from strata.notebook import cell_traceback

    text = cell_traceback._exited("the cell", "x" * 50_000)
    assert text.endswith("x" * 10 + "...")
    assert len(text) < 10_100


def test_an_interrupt_still_writes_the_result(harness_script):
    """Ctrl-C in the server's terminal reaches the harness too."""
    result = run_harness(harness_script, {"source": "raise KeyboardInterrupt", "inputs": {}})

    assert result["success"] is False
    assert result["error"] == "KeyboardInterrupt"


def test_an_exception_whose_str_raises_is_still_reported(harness_script):
    source = (
        "class Unreadable(Exception):\n"
        "    def __str__(self):\n"
        "        raise ValueError\n"
        "raise Unreadable()\n"
    )
    result = run_harness(harness_script, {"source": source, "inputs": {}})

    assert result["success"] is False
    assert result["error"] == "Unreadable: <exception str() failed>"


def test_a_failure_the_handler_cannot_report_still_writes_a_result(harness_script):
    """Should reporting the failure itself raise, the result still says so, with the
    cell's print trail."""
    source = (
        "import sys\n"
        "print('step 1 done')\n"
        "def broken(*args, **kwargs):\n"
        "    raise ValueError\n"
        "sys.modules['_nb_cell_traceback'].failure_fields = broken\n"
        "raise KeyError('x')\n"
    )
    result = run_harness(harness_script, {"source": source, "inputs": {}})

    assert result["success"] is False
    assert result["error"] == "the harness failed reporting the cell's failure"
    assert result["stdout"] == "step 1 done\n"


@pytest.mark.parametrize(
    ("base", "attribute"), [("FileNotFoundError", "filename"), ("NameError", "name")]
)
@pytest.mark.parametrize("raises", ["raise ValueError", "sys.exit(2)"])
def test_an_exception_whose_attributes_raise_is_its_own_error(
    harness_script, tmp_path, base, attribute, raises
):
    """The hints and the traceback read them: a failure is still reported as itself."""
    source = (
        "import sys\n"
        f"class Odd({base}):\n"
        "    @property\n"
        f"    def {attribute}(self):\n"
        f"        {raises}\n"
        "raise Odd('it broke')\n"
    )
    manifest = {"source": source, "inputs": {}, "notebook_dir": str(tmp_path)}
    result = run_harness(harness_script, manifest)

    assert result["success"] is False
    assert result["error"] == "Odd: it broke", result


def test_failure_fields_survive_a_traceback_that_cannot_be_formatted(monkeypatch):
    """The warm worker's own fallback and a batch's would lose the cell's error and
    console, so the report itself must not raise; the frames need nothing it read."""
    from strata.notebook import cell_traceback

    def unformattable(exc):
        raise SystemExit(2)

    monkeypatch.setattr(cell_traceback, "_from_the_cell", unformattable)
    source = "def helper():\n    raise KeyError('it broke')\nhelper()\n"
    try:
        exec(compile(source, cell_traceback.cell_filename("c1"), "exec"), {})
    except KeyError as exc:
        fields = cell_traceback.failure_fields(exc)

    assert fields["error"] == "KeyError: 'it broke'"
    tail = "KeyError: 'it broke'\n(its traceback could not be formatted)\n"
    # From the cell's first frame, as the trimmed traceback is; the full one keeps this test's.
    assert fields["traceback"] == (
        "Traceback (most recent call last):\n"
        '  File "<cell c1>", line 3, in <module>\n'
        '  File "<cell c1>", line 2, in helper\n' + tail
    ), fields["traceback"]
    # What formatted keeps its own text.
    assert "test_failure_fields_survive" in fields["traceback_full"]
    assert fields["traceback_full"].endswith("KeyError: 'it broke'\n"), fields["traceback_full"]


def test_safely_keeps_an_interrupt_and_nothing_else():
    import asyncio

    from strata.notebook import cell_traceback

    def raising(exc):
        def read():
            raise exc

        return read

    assert cell_traceback.safely(raising(asyncio.CancelledError()), "d") == "d"
    assert cell_traceback.safely(raising(SystemExit(2)), "d") == "d"
    with pytest.raises(KeyboardInterrupt):
        cell_traceback.safely(raising(KeyboardInterrupt()), "d")


def _odd_objects():
    import asyncio

    class Cancels:
        def __str__(self):
            raise asyncio.CancelledError

    class Unmeasurable(str):
        def __len__(self):
            raise ValueError

    class ReturnsOdd:
        def __str__(self):
            return Unmeasurable("odd")

    class NoClass:
        @property
        def __class__(self):
            raise ValueError

    class BadInt(int):
        def __int__(self):
            raise ValueError

    return Cancels, ReturnsOdd, NoClass, BadInt


def test_an_odd_exit_code_still_reads_as_an_exit():
    import sys

    from strata.notebook import cell_traceback

    cancels, returns_odd, no_class, bad_int = _odd_objects()
    stops = "which stops it before its end"
    for code, message in (
        (cancels(), f"the cell exited, {stops}: <str() failed>"),
        (returns_odd(), f"the cell exited, {stops}: odd"),
        (bad_int(3), f"the cell exited with status <int() failed>, {stops}"),
    ):
        with pytest.raises(RuntimeError) as raised, cell_traceback.exit_fails_the_cell():
            sys.exit(code)
        assert str(raised.value) == message, code
    # Its text, though its own __class__ raises.
    with pytest.raises(RuntimeError) as raised, cell_traceback.exit_fails_the_cell():
        sys.exit(no_class())
    assert str(raised.value).startswith(f"the cell exited, {stops}: <"), raised.value
    assert "NoClass object at" in str(raised.value), raised.value


def test_an_odd_exception_is_still_reported_and_gets_no_hint(tmp_path, monkeypatch):
    import asyncio

    from strata.notebook import cell_traceback, harness, work_dir

    def cancel(self):
        raise asyncio.CancelledError

    class Ambiguous:
        """As a numpy array compares: a result with no truth value."""

        def __eq__(self, other):
            return self

        def __bool__(self):
            raise ValueError

    class Cancels(Exception):
        __str__ = cancel

    class NoClass(Exception):
        @property
        def __class__(self):
            raise ValueError

    class FileCancels(FileNotFoundError):
        filename = property(cancel)

    class NotesTuple(FileNotFoundError):
        def add_note(self, note):
            raise TypeError

    class NameCancels(NameError):
        name = property(cancel)

    class NameAmbiguous(NameError):
        name = Ambiguous()

    (tmp_path / "data.csv").write_text("")
    work = tmp_path / "work"
    work.mkdir()
    manifest = {"notebook_dir": str(tmp_path)}

    assert cell_traceback.error_text(Cancels()) == "Cancels: <exception str() failed>"
    for exc in (NoClass(), FileCancels()):
        assert work_dir.hint(exc, manifest) is None, exc
    for exc in (NoClass(), NameCancels(), NameAmbiguous()):
        assert harness.no_client_hint(exc, {}) is None, exc
    # Notes that refuse one: the hint stands without its note.
    unnoted = NameError("x", name="strata")
    unnoted.__notes__ = ()
    assert harness.no_client_hint(unnoted, {}).startswith("A cell on a worker")
    monkeypatch.chdir(work)
    assert "read it as ../data.csv" in work_dir.hint(
        NotesTuple(2, "No such file", "data.csv"), manifest
    )
    # The plain error gets its hint.
    assert work_dir.hint(FileNotFoundError(2, "No such file", "data.csv"), manifest)


def _unreadable(self):
    raise ValueError


class _NoClass:
    __class__ = property(_unreadable)  # type: ignore[assignment]


@pytest.mark.parametrize(
    "odd",
    [
        {"code": property(_unreadable)},
        {"__cause__": property(_unreadable)},
        {"__cause__": property(lambda self: 5)},
        {"__traceback__": property(_unreadable)},
        {"__traceback__": property(lambda self: 5)},
        # isinstance would read these objects' own __class__.
        {"__cause__": property(lambda self: _NoClass())},
        {"__traceback__": property(lambda self: _NoClass())},
    ],
)
def test_an_odd_system_exit_still_reads_as_an_exit(odd):
    from strata.notebook import cell_traceback

    odd_exit = type("OddExit", (SystemExit,), odd)
    message = escaped = None
    try:
        with cell_traceback.exit_fails_the_cell():
            raise odd_exit
    except RuntimeError as exc:
        message = str(exc)
    except BaseException as exc:
        escaped = type(exc).__name__
    # Failed outside the handler, so pytest never formats the odd exit itself.
    assert escaped is None, escaped
    assert message == "the cell exited, which stops it before its end"


def test_a_traceback_with_no_frames_has_no_header(monkeypatch):
    from strata.notebook import cell_traceback

    def unformattable(exc):
        raise ValueError

    monkeypatch.setattr(cell_traceback, "_from_the_cell", unformattable)
    fields = cell_traceback.failure_fields(KeyError("x"))
    assert fields["traceback"] == "KeyError: 'x'\n(its traceback could not be formatted)\n"
    # What formatted keeps its own text.
    assert fields["traceback_full"] == "KeyError: 'x'\n"


def test_a_file_hint_survives_a_filename_that_raises(tmp_path):
    """Every harness asks it, the warm worker and a batch with no fallback of their own."""
    from strata.notebook import work_dir

    class Odd(FileNotFoundError):
        @property
        def filename(self):
            raise ValueError

    assert work_dir.hint(Odd(), {"notebook_dir": str(tmp_path)}) is None
