"""Tests for cell annotation parsing."""

from __future__ import annotations

from strata.notebook.annotations import parse_annotations


class TestNameAnnotation:
    """Tests for the @name annotation."""

    def test_name_with_spaces(self):
        """@name accepts human-readable names with spaces."""
        result = parse_annotations("# @name Load arXiv Papers\nx = 1")
        assert result.name == "Load arXiv Papers"

    def test_name_identifier(self):
        """@name accepts Python identifiers (backward compat for prompt cells)."""
        result = parse_annotations("# @name research_themes\n")
        assert result.name == "research_themes"

    def test_name_with_special_chars(self):
        """@name accepts names with parentheses and other chars."""
        result = parse_annotations("# @name Aggregate by Topic (DataFusion)\nx = 1")
        assert result.name == "Aggregate by Topic (DataFusion)"

    def test_name_empty_is_none(self):
        """Empty @name is ignored."""
        result = parse_annotations("# @name\nx = 1")
        assert result.name is None

    def test_name_no_annotation(self):
        """No @name annotation → None."""
        result = parse_annotations("x = 1")
        assert result.name is None

    def test_name_with_worker(self):
        """@name coexists with @worker."""
        result = parse_annotations("# @name Train Model\n# @worker gpu-fly\nx = 1")
        assert result.name == "Train Model"
        assert result.worker == "gpu-fly"

    def test_name_after_non_comment_ignored(self):
        """@name after the leading comment block is ignored."""
        result = parse_annotations("x = 1\n# @name Late Name\n")
        assert result.name is None


class TestNameInPromptAnalyzer:
    """Verify that prompt_analyzer still requires identifiers for @name."""

    def test_prompt_analyzer_requires_identifier(self):
        from strata.notebook.prompt_analyzer import analyze_prompt_cell

        result = analyze_prompt_cell("# @name research_themes\nHello {{ x }}")
        assert result.name == "research_themes"

    def test_prompt_analyzer_rejects_non_identifier(self):
        from strata.notebook.prompt_analyzer import analyze_prompt_cell

        result = analyze_prompt_cell("# @name Research Themes\nHello {{ x }}")
        # Non-identifier name is rejected — falls back to default "result"
        assert result.name == "result"


class TestNameInRoutes:
    """Verify that @name flows through to the API response."""

    def test_cell_annotations_include_name(self, tmp_path):
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        notebook_dir = create_notebook(tmp_path, "NameTest", initialize_environment=False)
        add_cell_to_notebook(notebook_dir, "c1")
        write_cell(notebook_dir, "c1", "# @name My Cool Cell\nx = 1")

        state = parse_notebook(notebook_dir)
        session = NotebookSession(state, notebook_dir)
        data = session.serialize_notebook_state()

        cell = next(c for c in data["cells"] if c["id"] == "c1")
        assert cell["annotations"]["name"] == "My Cool Cell"

    def test_cell_annotations_name_absent_when_not_set(self, tmp_path):
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        notebook_dir = create_notebook(tmp_path, "NoNameTest", initialize_environment=False)
        add_cell_to_notebook(notebook_dir, "c1")
        write_cell(notebook_dir, "c1", "x = 1")

        state = parse_notebook(notebook_dir)
        session = NotebookSession(state, notebook_dir)
        data = session.serialize_notebook_state()

        cell = next(c for c in data["cells"] if c["id"] == "c1")
        assert cell["annotations"]["name"] is None


class TestLoopAnnotation:
    """Tests for ``@loop`` / ``@loop_until`` parsing."""

    def test_loop_requires_max_iter_and_carry(self):
        """A well-formed ``@loop`` populates max_iter and carry."""
        result = parse_annotations("# @loop max_iter=10 carry=state\nstate = refine(state)")
        assert result.loop is not None
        assert result.loop.max_iter == 10
        assert result.loop.carry == "state"
        assert result.loop.until_expr is None
        assert result.loop.start_from_cell is None
        assert result.loop.start_from_iter is None

    def test_loop_until_is_captured(self):
        """``@loop_until`` attaches its expression to the LoopAnnotation."""
        source = (
            "# @loop max_iter=10 carry=state\n"
            "# @loop_until state['confidence'] > 0.9\n"
            "state = refine(state)\n"
        )
        result = parse_annotations(source)
        assert result.loop is not None
        assert result.loop.until_expr == "state['confidence'] > 0.9"

    def test_loop_start_from_parses_cell_and_iter(self):
        """``start_from=<cell>@iter=<k>`` is split into its two fields."""
        source = "# @loop max_iter=5 carry=state start_from=evolve@iter=3\nstate = propose(state)"
        result = parse_annotations(source)
        assert result.loop is not None
        assert result.loop.start_from_cell == "evolve"
        assert result.loop.start_from_iter == 3

    def test_loop_merges_multiple_lines(self):
        """Two ``@loop`` lines and a trailing ``@loop_until`` all merge."""
        source = (
            "# @loop max_iter=20\n"
            "# @loop carry=state\n"
            "# @loop_until state.get('done')\n"
            "state = tick(state)\n"
        )
        result = parse_annotations(source)
        assert result.loop is not None
        assert result.loop.max_iter == 20
        assert result.loop.carry == "state"
        assert result.loop.until_expr == "state.get('done')"

    def test_loop_absent_when_no_annotation(self):
        result = parse_annotations("x = 1")
        assert result.loop is None

    def test_loop_ignores_unknown_keys(self):
        """Unknown ``key=value`` pairs on a ``@loop`` line are silently skipped."""
        result = parse_annotations(
            "# @loop max_iter=3 carry=state nonsense=42\nstate = tick(state)"
        )
        assert result.loop is not None
        assert result.loop.max_iter == 3
        assert result.loop.carry == "state"

    def test_loop_start_from_malformed_is_dropped(self):
        """A ``start_from`` value that does not match ``<cell>@iter=<int>`` is dropped."""
        result = parse_annotations("# @loop max_iter=5 carry=state start_from=badvalue\n")
        assert result.loop is not None
        assert result.loop.start_from_cell is None
        assert result.loop.start_from_iter is None

    def test_loop_until_without_loop_still_captures_expr(self):
        """``@loop_until`` alone should still record the expression so validation can
        surface the missing ``max_iter``/``carry`` as errors."""
        result = parse_annotations("# @loop_until x > 0\nx = 1")
        assert result.loop is not None
        assert result.loop.until_expr == "x > 0"
        assert result.loop.max_iter == 0
        assert result.loop.carry == ""

    def test_loop_annotation_surfaces_in_cell_serialization(self, tmp_path):
        """``session.serialize_cell`` must include the loop annotation so
        the frontend can populate the iteration picker without a second
        round-trip to parse the cell source itself."""
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        notebook_dir = create_notebook(tmp_path, "LoopAnnotTest", initialize_environment=False)
        add_cell_to_notebook(notebook_dir, "c1")
        write_cell(
            notebook_dir,
            "c1",
            (
                "# @loop max_iter=5 carry=state start_from=donor@iter=2\n"
                "# @loop_until state['done']\n"
                "state = step(state)\n"
            ),
        )

        state = parse_notebook(notebook_dir)
        session = NotebookSession(state, notebook_dir)
        data = session.serialize_notebook_state()

        cell = next(c for c in data["cells"] if c["id"] == "c1")
        loop_payload = cell["annotations"]["loop"]
        assert loop_payload is not None
        assert loop_payload["max_iter"] == 5
        assert loop_payload["carry"] == "state"
        assert loop_payload["until_expr"] == "state['done']"
        assert loop_payload["start_from_cell"] == "donor"
        assert loop_payload["start_from_iter"] == 2

    def test_cell_annotations_loop_none_for_regular_cell(self, tmp_path):
        """Regular (non-loop) cells must emit ``loop: None`` so the
        frontend's annotation parser doesn't trip on a missing key."""
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        notebook_dir = create_notebook(tmp_path, "NoLoopTest", initialize_environment=False)
        add_cell_to_notebook(notebook_dir, "c1")
        write_cell(notebook_dir, "c1", "x = 1")

        state = parse_notebook(notebook_dir)
        session = NotebookSession(state, notebook_dir)
        data = session.serialize_notebook_state()

        cell = next(c for c in data["cells"] if c["id"] == "c1")
        assert cell["annotations"]["loop"] is None


class TestVariantAnnotation:
    """Tests for the @variant grouping annotation."""

    def test_variant_parses_group_and_name(self):
        result = parse_annotations("# @variant model_choice gpt4\nx = 1")
        assert result.variant is not None
        assert result.variant.group == "model_choice"
        assert result.variant.name == "gpt4"

    def test_variant_absent_when_not_set(self):
        result = parse_annotations("x = 1")
        assert result.variant is None

    def test_variant_missing_name_is_dropped(self):
        result = parse_annotations("# @variant model_choice\nx = 1")
        assert result.variant is None

    def test_variant_extra_tokens_dropped(self):
        result = parse_annotations("# @variant model_choice gpt4 extra\nx = 1")
        assert result.variant is None

    def test_variant_non_identifier_group_dropped(self):
        result = parse_annotations("# @variant model-choice gpt4\nx = 1")
        assert result.variant is None

    def test_variant_non_identifier_name_dropped(self):
        result = parse_annotations("# @variant model_choice gpt-4\nx = 1")
        assert result.variant is None

    def test_variant_coexists_with_other_annotations(self):
        source = "# @name GPT-4 model\n# @variant model_choice gpt4\nx = 1"
        result = parse_annotations(source)
        assert result.variant is not None
        assert result.variant.group == "model_choice"
        assert result.variant.name == "gpt4"
        assert result.name == "GPT-4 model"


class TestAfterAnnotation:
    """Tests for the @after ordering-dependency annotation."""

    def test_after_single_cell(self):
        result = parse_annotations("# @after setup\nx = 1")
        assert result.after == ["setup"]

    def test_after_multiple_lines_stack(self):
        result = parse_annotations("# @after setup\n# @after seed_db\nx = 1")
        assert result.after == ["setup", "seed_db"]

    def test_after_multiple_ids_on_one_line(self):
        result = parse_annotations("# @after setup seed_db\nx = 1")
        assert result.after == ["setup", "seed_db"]

    def test_after_dedupes_repeated_ids(self):
        result = parse_annotations("# @after setup\n# @after setup\nx = 1")
        assert result.after == ["setup"]

    def test_after_empty_when_absent(self):
        result = parse_annotations("x = 1")
        assert result.after == []


class TestTableAnnotation:
    """Parsing of the table input annotation."""

    def test_basic_table(self):
        result = parse_annotations("# @table trips file:///data/warehouse#nyc.trips\nx = 1\n")
        assert len(result.tables) == 1
        table = result.tables[0]
        assert table.name == "trips"
        assert table.uri == "file:///data/warehouse#nyc.trips"
        assert table.snapshot_pin is None

    def test_snapshot_pin(self):
        result = parse_annotations(
            "# @table events s3://bucket/wh#db.events snapshot=1292033279574548405\nx = 1\n"
        )
        assert result.tables[0].snapshot_pin == 1292033279574548405

    def test_invalid_name_ignored(self):
        result = parse_annotations("# @table 2bad file:///wh#db.t\nx = 1\n")
        assert result.tables == []

    def test_missing_uri_ignored(self):
        result = parse_annotations("# @table trips\nx = 1\n")
        assert result.tables == []

    def test_malformed_snapshot_ignored(self):
        result = parse_annotations("# @table trips file:///wh#db.t snapshot=abc\nx = 1\n")
        assert result.tables == []

    def test_multiple_tables(self):
        result = parse_annotations(
            "# @table a file:///wh#db.a\n# @table b file:///wh#db.b\nx = 1\n"
        )
        assert [t.name for t in result.tables] == ["a", "b"]

    def test_wire_payload_includes_tables(self):
        result = parse_annotations("# @table trips file:///wh#db.t\nx = 1\n")
        payload = result.to_wire_payload()
        assert payload["tables"] == [
            {"name": "trips", "uri": "file:///wh#db.t", "snapshot_pin": None}
        ]


class TestSpliceDirectives:
    """Tests for set_annotation_directive / remove_annotation_directive."""

    def _set(self, *args):
        from strata.notebook.annotations import set_annotation_directive

        return set_annotation_directive(*args)

    def _rm(self, *args):
        from strata.notebook.annotations import remove_annotation_directive

        return remove_annotation_directive(*args)

    def test_set_inserts_on_bare_cell(self):
        out = self._set("x = 1\n", "worker", "gpu-box")
        assert out == "# @worker gpu-box\nx = 1\n"
        assert parse_annotations(out).worker == "gpu-box"

    def test_set_groups_then_replaces_in_place(self):
        out = self._set("x = 1\n", "worker", "gpu-box")
        out = self._set(out, "name", "feat")  # inserted after the existing directive
        assert out == "# @worker gpu-box\n# @name feat\nx = 1\n"
        out = self._set(out, "worker", "cpu")  # replaced, not appended
        assert out == "# @worker cpu\n# @name feat\nx = 1\n"

    def test_set_preserves_body_and_plain_comments(self):
        out = self._set("# header\nimport os\nx = 1\n", "name", "n")
        assert "# header" in out and "import os" in out
        assert parse_annotations(out).name == "n"

    def test_set_empty_value_renders_flag(self):
        assert self._set("x = 1\n", "output", "") == "# @output\nx = 1\n"

    def test_set_collapses_duplicate_directives(self):
        dup = "# @worker a\n# @worker b\nx = 1\n"
        assert self._set(dup, "worker", "c") == "# @worker c\nx = 1\n"

    def test_remove_drops_directive_keeps_rest(self):
        src = "# @worker gpu\n# @name feat\nx = 1\n"
        out = self._rm(src, "worker")
        assert out == "# @name feat\nx = 1\n"
        assert parse_annotations(out).worker is None

    def test_remove_absent_key_is_noop(self):
        assert self._rm("# @name feat\nx = 1\n", "worker") == "# @name feat\nx = 1\n"

    def test_no_trailing_newline_preserved(self):
        assert self._set("x = 1", "name", "n") == "# @name n\nx = 1"
