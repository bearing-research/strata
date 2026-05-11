"""Tests for cell annotation cross-reference validation.

Parsing lives in ``test_annotations.py``. This file covers the five
diagnostic codes produced by ``validate_cell_annotations`` — the
cross-reference checks that fire on notebook open, reload, and WS
source flush (but never during active typing).
"""

from __future__ import annotations

import pytest

from strata.notebook.annotation_validation import validate_cell_annotations
from strata.notebook.models import (
    CellState,
    MountMode,
    MountSpec,
    NotebookState,
    WorkerBackendType,
    WorkerSpec,
)


def _nb(*, workers: list[WorkerSpec] | None = None, mounts: list[MountSpec] | None = None):
    """Build a minimal NotebookState for validation scenarios."""
    return NotebookState(
        id="test-nb",
        name="test",
        workers=workers or [],
        mounts=mounts or [],
    )


def _cell(source: str, cell_id: str = "c1") -> CellState:
    return CellState(id=cell_id, source=source)


def _codes(cell: CellState, nb: NotebookState) -> list[str]:
    return [d.code for d in validate_cell_annotations(cell, nb)]


class TestNoDiagnostics:
    """A clean cell with valid annotations should emit no diagnostics."""

    def test_empty_cell(self):
        assert _codes(_cell(""), _nb()) == []

    def test_cell_without_annotations(self):
        assert _codes(_cell("x = 1"), _nb()) == []

    def test_valid_builtin_worker(self):
        cell = _cell("# @worker local\nx = 1")
        assert _codes(cell, _nb()) == []

    def test_valid_declared_worker(self):
        nb = _nb(workers=[WorkerSpec(name="gpu-a100", backend=WorkerBackendType.EXECUTOR)])
        cell = _cell("# @worker gpu-a100\nx = 1")
        assert _codes(cell, nb) == []

    def test_valid_mount_uri(self):
        cell = _cell("# @mount data s3://bucket/prefix ro\nx = 1")
        assert _codes(cell, _nb()) == []

    def test_valid_timeout(self):
        cell = _cell("# @timeout 30\nx = 1")
        assert _codes(cell, _nb()) == []

    def test_valid_env(self):
        cell = _cell("# @env APP_MODE=prod\nx = 1")
        assert _codes(cell, _nb()) == []


class TestWorkerUnknown:
    """`@worker <name>` where <name> isn't in the notebook worker catalog."""

    def test_unknown_worker_name(self):
        cell = _cell("# @worker fake-worker\nx = 1")
        assert _codes(cell, _nb()) == ["worker_unknown"]

    def test_unknown_worker_with_other_workers_declared(self):
        nb = _nb(workers=[WorkerSpec(name="real-worker", backend=WorkerBackendType.EXECUTOR)])
        cell = _cell("# @worker different-worker\nx = 1")
        assert _codes(cell, nb) == ["worker_unknown"]

    def test_diagnostic_carries_line_number(self):
        cell = _cell("# @worker ghost\nx = 1")
        diagnostics = validate_cell_annotations(cell, _nb())
        assert len(diagnostics) == 1
        assert diagnostics[0].line == 1

    def test_diagnostic_message_mentions_name(self):
        cell = _cell("# @worker ghost\nx = 1")
        diagnostics = validate_cell_annotations(cell, _nb())
        assert "ghost" in diagnostics[0].message


class TestMountUriUnsupported:
    """`@mount` URI scheme not in the supported set."""

    @pytest.mark.parametrize(
        "uri",
        ["ftp://bad/path", "http://bad/path", "notascheme"],
    )
    def test_unsupported_scheme(self, uri):
        cell = _cell(f"# @mount data {uri} ro\nx = 1")
        assert "mount_uri_unsupported" in _codes(cell, _nb())

    def test_supported_schemes_pass(self):
        for uri in [
            "file:///tmp/x",
            "s3://bucket/prefix",
            "gs://bucket/prefix",
            "gcs://bucket/prefix",
            "az://container/prefix",
            "azure://container/prefix",
        ]:
            cell = _cell(f"# @mount data {uri} ro\nx = 1")
            assert "mount_uri_unsupported" not in _codes(cell, _nb())


class TestMountShadowsNotebook:
    """`@mount <name>` where <name> is already a notebook-level mount."""

    def test_shadowing_existing_mount_emits_info(self):
        nb = _nb(
            mounts=[MountSpec(name="data", uri="s3://nb/path", mode=MountMode.READ_ONLY)],
        )
        cell = _cell("# @mount data s3://cell/path ro\nx = 1")
        diagnostics = validate_cell_annotations(cell, nb)
        codes = [d.code for d in diagnostics]
        assert "mount_shadows_notebook" in codes
        shadow = next(d for d in diagnostics if d.code == "mount_shadows_notebook")
        assert shadow.severity == "info"

    def test_unique_mount_name_not_shadowing(self):
        nb = _nb(
            mounts=[MountSpec(name="data", uri="s3://nb/path", mode=MountMode.READ_ONLY)],
        )
        cell = _cell("# @mount scratch s3://cell/path ro\nx = 1")
        assert "mount_shadows_notebook" not in _codes(cell, nb)


class TestTimeoutNotNumeric:
    """`@timeout` value malformed, missing, non-numeric, or <= 0."""

    def test_non_numeric_value(self):
        cell = _cell("# @timeout abc\nx = 1")
        assert _codes(cell, _nb()) == ["timeout_not_numeric"]

    def test_missing_value(self):
        cell = _cell("# @timeout\nx = 1")
        assert _codes(cell, _nb()) == ["timeout_not_numeric"]

    def test_zero(self):
        cell = _cell("# @timeout 0\nx = 1")
        assert _codes(cell, _nb()) == ["timeout_not_numeric"]

    def test_negative(self):
        cell = _cell("# @timeout -5\nx = 1")
        assert _codes(cell, _nb()) == ["timeout_not_numeric"]

    def test_positive_float(self):
        cell = _cell("# @timeout 3.5\nx = 1")
        assert _codes(cell, _nb()) == []

    def test_diagnostic_carries_line_number(self):
        # Line 2 because line 1 is a different annotation
        cell = _cell("# @worker local\n# @timeout abc\nx = 1")
        diagnostics = validate_cell_annotations(cell, _nb())
        assert any(d.code == "timeout_not_numeric" and d.line == 2 for d in diagnostics)


class TestEnvMalformed:
    """`@env` missing the `KEY=value` format."""

    def test_missing_equals(self):
        cell = _cell("# @env JUST_KEY\nx = 1")
        assert _codes(cell, _nb()) == ["env_malformed"]

    def test_leading_equals_no_key(self):
        cell = _cell("# @env =value\nx = 1")
        assert _codes(cell, _nb()) == ["env_malformed"]

    def test_empty_value_is_allowed(self):
        # `KEY=` with an empty value is a common pattern for clearing
        # an env var; parser accepts it and validator shouldn't flag.
        cell = _cell("# @env CLEAR_ME=\nx = 1")
        assert _codes(cell, _nb()) == []

    def test_value_with_equals_signs(self):
        # Values may contain `=` (e.g. base64, signed URLs).
        cell = _cell("# @env KEY=part1=part2=part3\nx = 1")
        assert _codes(cell, _nb()) == []


class TestMultipleDiagnostics:
    """A cell with multiple issues should report all of them."""

    def test_worker_and_timeout_both_bad(self):
        cell = _cell("# @worker ghost\n# @timeout abc\nx = 1")
        codes = _codes(cell, _nb())
        assert "worker_unknown" in codes
        assert "timeout_not_numeric" in codes

    def test_mount_unsupported_and_shadowing_coexist(self):
        nb = _nb(mounts=[MountSpec(name="data", uri="s3://nb/p", mode=MountMode.READ_ONLY)])
        # Same name as notebook mount + unsupported scheme
        cell = _cell("# @mount data ftp://bad/p ro\nx = 1")
        codes = _codes(cell, nb)
        assert "mount_uri_unsupported" in codes
        assert "mount_shadows_notebook" in codes


class TestAnnotationBlockBoundary:
    """Validation only scans the leading comment block."""

    def test_annotations_after_code_ignored(self):
        # The parser stops at the first non-comment line, so a malformed
        # annotation after code shouldn't produce a diagnostic.
        cell = _cell("x = 1\n# @timeout abc")
        assert _codes(cell, _nb()) == []

    def test_blank_lines_inside_comment_block_allowed(self):
        cell = _cell("# @worker local\n\n# @timeout 30\nx = 1")
        assert _codes(cell, _nb()) == []


class TestLoopAnnotationDiagnostics:
    """Validation of ``@loop`` / ``@loop_until`` cross-references."""

    def test_valid_loop_passes(self):
        source = (
            "# @loop max_iter=10 carry=state\n# @loop_until state['done']\nstate = refine(state)\n"
        )
        cell = _cell(source)
        cell.defines = ["state"]
        assert _codes(cell, _nb()) == []

    def test_missing_max_iter_is_error(self):
        source = "# @loop carry=state\nstate = refine(state)\n"
        cell = _cell(source)
        cell.defines = ["state"]
        codes = _codes(cell, _nb())
        assert "loop_missing_max_iter" in codes

    def test_loop_until_without_max_iter_triggers_missing_max_iter(self):
        source = "# @loop_until x > 0\nx = 1\n"
        cell = _cell(source)
        cell.defines = ["x"]
        codes = _codes(cell, _nb())
        assert "loop_missing_max_iter" in codes
        assert "loop_missing_carry" in codes

    def test_missing_carry_is_error(self):
        source = "# @loop max_iter=10\nstate = refine(state)\n"
        cell = _cell(source)
        cell.defines = ["state"]
        codes = _codes(cell, _nb())
        assert "loop_missing_carry" in codes

    def test_carry_not_defined_warns(self):
        source = "# @loop max_iter=5 carry=state\nvalue = 1\n"
        cell = _cell(source)
        cell.defines = ["value"]
        codes = _codes(cell, _nb())
        assert "loop_carry_unknown" in codes

    def test_until_syntax_error(self):
        source = (
            "# @loop max_iter=5 carry=state\n"
            "# @loop_until not a valid = expression\n"
            "state = tick(state)\n"
        )
        cell = _cell(source)
        cell.defines = ["state"]
        codes = _codes(cell, _nb())
        assert "loop_until_syntax_error" in codes

    def test_start_from_unknown_cell(self):
        source = "# @loop max_iter=5 carry=state start_from=ghost@iter=3\nstate = tick(state)\n"
        cell = _cell(source)
        cell.defines = ["state"]
        codes = _codes(cell, _nb())
        assert "loop_start_from_unknown" in codes

    def test_start_from_known_cell_passes(self):
        source = "# @loop max_iter=5 carry=state start_from=donor@iter=2\nstate = tick(state)\n"
        cell = _cell(source, cell_id="c1")
        cell.defines = ["state"]
        donor = _cell("# @loop max_iter=20 carry=state\nstate = tick(state)", cell_id="donor")
        nb = NotebookState(id="test-nb", name="test", cells=[donor, cell])
        codes = _codes(cell, nb)
        assert "loop_start_from_unknown" not in codes

    def test_start_from_self_is_error(self):
        source = "# @loop max_iter=5 carry=state start_from=c1@iter=0\nstate = tick(state)\n"
        cell = _cell(source, cell_id="c1")
        cell.defines = ["state"]
        nb = NotebookState(id="test-nb", name="test", cells=[cell])
        codes = _codes(cell, nb)
        assert "loop_start_from_unknown" in codes


class TestModuleExportBlockedDiagnostic:
    """Warn when a Python cell mixes defs/classes with non-literal
    runtime logic, since the def/class can't cross the module-export
    boundary downstream."""

    def test_pure_module_cell_is_silent(self):
        cell = _cell("import math\n\ndef add(x, y):\n    return x + y\n")
        assert _codes(cell, _nb()) == []

    def test_literal_constants_with_defs_is_silent(self):
        """The main UX fix — ``STEP = 0.5`` alongside a def should no
        longer warn, because literal constants are now exportable."""
        cell = _cell("STEP = 0.5\n\ndef scale(x):\n    return x * STEP\n")
        assert _codes(cell, _nb()) == []

    def test_def_with_unresolved_runtime_dep_warns_when_consumed(self):
        """``scale`` closes over ``STEP``, which is computed at runtime
        and thus dropped from the slice. The synthetic module would
        NameError when ``scale`` is called downstream — surface the
        problem at validation time so the user can fix it before
        execution."""
        cell = _cell("STEP = compute_step()\n\ndef scale(x):\n    return x * STEP\n")
        downstream = _cell("y = scale(2)", cell_id="c2")
        downstream.references = ["scale"]
        nb = _nb()
        nb.cells = [cell, downstream]
        diags = [
            d for d in validate_cell_annotations(cell, nb) if d.code == "module_export_blocked"
        ]
        assert len(diags) == 1
        message = diags[0].message
        assert "`scale`" in message
        # The diagnostic now names the unresolved variable so the user
        # knows exactly what to move or import.
        assert "STEP" in message

    def test_top_level_expression_alongside_self_contained_class_is_silent(self):
        """Slicing drops the top-level expression; ``Config``'s body
        only references literals/builtins, so the slice exports it
        cleanly. No diagnostic — this is one of the user-facing wins of
        slicing."""
        cell = _cell("print('hi')\n\nclass Config:\n    debug = True\n")
        downstream = _cell("c = Config()", cell_id="c2")
        downstream.references = ["Config"]
        nb = _nb()
        nb.cells = [cell, downstream]
        codes = [d.code for d in validate_cell_annotations(cell, nb)]
        assert "module_export_blocked" not in codes

    def test_class_with_unresolved_base_warns_when_consumed(self):
        """A base class whose name isn't bound in the slice is a real
        export blocker — class body executes at module load and would
        NameError on the missing base."""
        cell = _cell("Parent = make_parent()\n\nclass Child(Parent):\n    pass\n")
        downstream = _cell("c = Child()", cell_id="c2")
        downstream.references = ["Child"]
        nb = _nb()
        nb.cells = [cell, downstream]
        diags = [
            d for d in validate_cell_annotations(cell, nb) if d.code == "module_export_blocked"
        ]
        assert len(diags) == 1
        assert "Parent" in diags[0].message

    def test_unused_helper_def_is_silent(self):
        """Cells with private helpers nobody else references shouldn't warn.

        Common in benchmark scoring loops, ad-hoc parsing — the def is a
        local helper, not an export. Warning every such cell drowns the
        signal users actually care about.
        """
        cell = _cell("def parse(raw):\n    return raw.strip()\n\nx = parse('abc')\n")
        nb = _nb()
        nb.cells = [cell]
        codes = [d.code for d in validate_cell_annotations(cell, nb)]
        assert "module_export_blocked" not in codes

    def test_pure_runtime_cell_is_silent(self):
        """No defs/classes → nothing to warn about even if runtime."""
        cell = _cell("x = compute()\ny = x + 1\n")
        codes = [d.code for d in validate_cell_annotations(cell, _nb())]
        assert "module_export_blocked" not in codes


class TestVariantDiagnostics:
    """``# @variant`` validation: malformed lines, sibling contract, toml drift."""

    def _group_nb(self, cells: list[CellState], selections: dict[str, str] | None = None):
        nb = _nb()
        nb.cells = cells
        if selections:
            nb.variant_active_selections = dict(selections)
        return nb

    def test_well_formed_solo_variant_no_diagnostic(self):
        cell = _cell(
            "# @variant model gpt4\npreds = run()\n",
            cell_id="c1",
        )
        cell.defines = ["preds"]
        cell.variant_group = "model"
        cell.variant_name = "gpt4"
        nb = self._group_nb([cell])
        assert _codes(cell, nb) == []

    def test_malformed_variant_line(self):
        cell = _cell(
            "# @variant model\npreds = run()\n",
            cell_id="c1",
        )
        codes = _codes(cell, nb := _nb())
        assert "variant_malformed" in codes
        assert nb is not None  # silence unused var warning

    def test_contract_mismatch_surfaces_on_outlier(self):
        a = _cell(
            "# @variant model gpt4\npreds = 1\nextra = 2\n",
            cell_id="a",
        )
        a.defines = ["preds", "extra"]
        a.variant_group = "model"
        a.variant_name = "gpt4"

        b = _cell(
            "# @variant model claude\npreds = 1\n",
            cell_id="b",
        )
        b.defines = ["preds"]
        b.variant_group = "model"
        b.variant_name = "claude"

        nb = self._group_nb([a, b])
        # Outlier a (defines extra) gets the diagnostic when compared to b
        assert "variant_contract_mismatch" in _codes(a, nb)

    def test_active_unknown_when_toml_names_missing_variant(self):
        a = _cell(
            "# @variant model gpt4\npreds = 1\n",
            cell_id="a",
        )
        a.defines = ["preds"]
        a.variant_group = "model"
        a.variant_name = "gpt4"

        nb = self._group_nb([a], selections={"model": "ghost"})
        assert "variant_active_unknown" in _codes(a, nb)

    def test_active_known_emits_no_diagnostic(self):
        a = _cell(
            "# @variant model gpt4\npreds = 1\n",
            cell_id="a",
        )
        a.defines = ["preds"]
        a.variant_group = "model"
        a.variant_name = "gpt4"

        nb = self._group_nb([a], selections={"model": "gpt4"})
        codes = _codes(a, nb)
        assert "variant_active_unknown" not in codes


class TestPromptOutputSchemaDiagnostics:
    """Prompt cells surface malformed ``@output_schema`` as a warning."""

    def _prompt_cell(self, source: str) -> CellState:
        return CellState(id="c1", source=source, language="prompt")

    def test_valid_schema_no_diagnostic(self):
        cell = self._prompt_cell('# @output_schema {"type": "object"}\nSummarize {{ df }}')
        diags = validate_cell_annotations(cell, _nb())
        assert [d.code for d in diags] == []

    def test_invalid_json_surfaces_warning(self):
        cell = self._prompt_cell("# @output_schema {nope}\nHi")
        diags = validate_cell_annotations(cell, _nb())
        codes = [d.code for d in diags]
        assert codes == ["prompt_output_schema_invalid"]
        assert diags[0].severity == "warn"

    def test_non_object_schema_surfaces_warning(self):
        cell = self._prompt_cell("# @output_schema [1,2]\nHi")
        diags = validate_cell_annotations(cell, _nb())
        assert [d.code for d in diags] == ["prompt_output_schema_invalid"]

    def test_python_cell_does_not_run_prompt_validator(self):
        """A Python cell with text that looks like @output_schema stays silent."""
        cell = _cell("# @output_schema {bad\nx = 1")
        codes = _codes(cell, _nb())
        assert "prompt_output_schema_invalid" not in codes
