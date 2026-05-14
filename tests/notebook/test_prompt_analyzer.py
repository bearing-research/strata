"""Tests for prompt cell analyzer."""

from strata.notebook.prompt_analyzer import analyze_prompt_cell


class TestPromptAnalyzer:
    """Tests for {{ var }} extraction and annotation parsing."""

    def test_basic_reference(self):
        result = analyze_prompt_cell("Summarize {{ df }}")
        assert result.references == ["df"]
        assert result.defines == ["result"]
        assert result.template_body == "Summarize {{ df }}"

    def test_multiple_references(self):
        result = analyze_prompt_cell("Compare {{ df1 }} with {{ df2 }}")
        assert result.references == ["df1", "df2"]

    def test_no_duplicates(self):
        result = analyze_prompt_cell("{{ df }} and {{ df }}")
        assert result.references == ["df"]

    def test_attribute_access(self):
        result = analyze_prompt_cell("Show {{ df.describe() }}")
        assert result.references == ["df"]

    def test_name_annotation(self):
        result = analyze_prompt_cell("# @name summary\nSummarize {{ df }}")
        assert result.name == "summary"
        assert result.defines == ["summary"]
        assert result.references == ["df"]

    def test_model_annotation(self):
        result = analyze_prompt_cell("# @model gpt-4o\nHello")
        assert result.model == "gpt-4o"

    def test_temperature_annotation(self):
        result = analyze_prompt_cell("# @temperature 0.7\nHello")
        assert result.temperature == 0.7

    def test_output_schema_valid(self):
        source = (
            '# @output_schema {"type": "object", "properties": {"n": {"type": "integer"}}}\n'
            "Count items in {{ df }}"
        )
        result = analyze_prompt_cell(source)
        assert result.output_schema == {
            "type": "object",
            "properties": {"n": {"type": "integer"}},
        }
        assert result.output_schema_error is None

    def test_output_schema_invalid_json_reports_error(self):
        source = "# @output_schema {type: object}\nHi"
        result = analyze_prompt_cell(source)
        assert result.output_schema is None
        assert result.output_schema_error is not None
        assert "Invalid JSON" in result.output_schema_error

    def test_output_schema_must_be_object(self):
        result = analyze_prompt_cell("# @output_schema [1, 2, 3]\nHi")
        assert result.output_schema is None
        assert result.output_schema_error == "@output_schema must be a JSON object"

    def test_validate_retries_annotation(self):
        result = analyze_prompt_cell("# @validate_retries 5\nHi")
        assert result.validate_retries == 5

    def test_validate_retries_non_numeric_ignored(self):
        result = analyze_prompt_cell("# @validate_retries nope\nHi")
        assert result.validate_retries is None

    def test_validate_retries_zero_ignored(self):
        """A value of 0 would disable the first call entirely — ignore it."""
        result = analyze_prompt_cell("# @validate_retries 0\nHi")
        assert result.validate_retries is None

    def test_output_annotation(self):
        result = analyze_prompt_cell("# @output json\nExtract {{ text }}")
        assert result.output_type == "json"

    def test_system_annotation(self):
        result = analyze_prompt_cell("# @system You are a data analyst.\nAnalyze {{ df }}")
        assert result.system_prompt == "You are a data analyst."

    def test_max_tokens_annotation(self):
        result = analyze_prompt_cell("# @max_tokens 8192\nHello")
        assert result.max_tokens == 8192

    def test_all_annotations(self):
        source = """# @name entities
# @model claude-sonnet-4-20250514
# @temperature 0.0
# @output json
# @max_tokens 4096
# @system You extract entities from text.
Extract entities from {{ text }} as a JSON array."""
        result = analyze_prompt_cell(source)
        assert result.name == "entities"
        assert result.model == "claude-sonnet-4-20250514"
        assert result.temperature == 0.0
        assert result.output_type == "json"
        assert result.max_tokens == 4096
        assert result.system_prompt == "You extract entities from text."
        assert result.references == ["text"]
        assert result.defines == ["entities"]
        assert "Extract entities" in result.template_body

    def test_empty_source(self):
        result = analyze_prompt_cell("")
        assert result.references == []
        assert result.defines == ["result"]
        assert result.template_body == ""

    def test_no_references(self):
        result = analyze_prompt_cell("What is the meaning of life?")
        assert result.references == []

    def test_builtins_not_referenced(self):
        result = analyze_prompt_cell("Print {{ len }} of {{ data }}")
        assert "len" not in result.references
        assert "data" in result.references

    def test_invalid_name_ignored(self):
        result = analyze_prompt_cell("# @name 123invalid\nHello")
        assert result.name == "result"

    def test_default_name_is_result(self):
        result = analyze_prompt_cell("Hello")
        assert result.name == "result"
        assert result.defines == ["result"]


class TestShadowWarning:
    """Tests for DAG shadow detection."""

    def test_shadow_warning_on_duplicate_defines(self):
        from strata.notebook.dag import CellAnalysisWithId, NotebookDag

        cells = [
            CellAnalysisWithId(id="c1", defines=["result"], references=[]),
            CellAnalysisWithId(id="c2", defines=["result"], references=[]),
        ]
        dag = NotebookDag.from_cells(cells)
        assert "c2" in dag.shadow_warnings
        assert any("result" in w for w in dag.shadow_warnings["c2"])

    def test_no_warning_without_shadow(self):
        from strata.notebook.dag import CellAnalysisWithId, NotebookDag

        cells = [
            CellAnalysisWithId(id="c1", defines=["x"], references=[]),
            CellAnalysisWithId(id="c2", defines=["y"], references=["x"]),
        ]
        dag = NotebookDag.from_cells(cells)
        assert len(dag.shadow_warnings) == 0
