"""Executor for prompt-type notebook cells.

Resolves upstream artifacts to text, renders the prompt template,
calls the LLM, and stores the response as an artifact.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from strata.notebook.llm import (
    LlmConfig,
    chat_completion,
    estimate_tokens,
    render_prompt_template,
)
from strata.notebook.prompt_analyzer import analyze_prompt_cell
from strata.notebook.provenance import derive_subkey

if TYPE_CHECKING:
    from strata.notebook.session import NotebookSession

logger = logging.getLogger(__name__)

# Default attempt count: one call + two retries. Every attempt can
# consume tokens, so keep the ceiling low; users who need more can set
# ``# @validate_retries N``.
DEFAULT_VALIDATE_ATTEMPTS = 3


def _validation_errors(content: str, schema: dict[str, Any]) -> list[str]:
    """Return a list of human-readable, path-addressed validation errors.

    Empty list ⇔ the response is valid JSON *and* matches the schema.
    A single "not valid JSON" error collapses the JSON-parse failure
    into the same return type as schema failures so the caller has one
    thing to handle.
    """
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return [f"response is not valid JSON: {exc.msg} (line {exc.lineno}, col {exc.colno})"]

    try:
        import jsonschema
    except ImportError:
        # No validator available — fall back to "is it JSON".
        return []

    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(parsed), key=lambda e: list(e.absolute_path))
    if not errors:
        return []
    return [_format_one_error(err) for err in errors]


def _format_one_error(err: Any) -> str:
    """Render a jsonschema ValidationError as 'path: message'.

    Uses JSON Pointer style for the path (leading ``/``, empty pointer
    for the root). Matches what language models respond to best — they
    trained on JSON Schema docs that use this convention.
    """
    pointer = "/" + "/".join(str(part) for part in err.absolute_path) if err.absolute_path else ""
    return f"{pointer or '(root)'}: {err.message}"


def _format_retry_prompt(errors: list[str]) -> str:
    """Build the user-turn feedback message for a validation retry.

    Kept spare on purpose — the errors themselves are descriptive; any
    extra framing we add just burns tokens and risks the model
    over-interpreting our instructions over the schema.
    """
    bullets = "\n".join(f"- {err}" for err in errors)
    return (
        "Your previous response did not validate against the required schema.\n"
        f"Errors:\n{bullets}\n"
        "Return a corrected JSON object that satisfies the schema."
    )


def compute_prompt_provenance_hash(
    *,
    rendered: str,
    model: str,
    temperature: float,
    system_prompt: str | None,
    output_type: str,
    output_schema: dict[str, Any] | None,
) -> str:
    """Stable cache key for a prompt-cell invocation.

    Includes the schema fingerprint so editing ``@output_schema``
    invalidates cached responses even when the template body and model
    params are unchanged — a schema change means the user wants a
    differently-shaped answer.
    """
    schema_fp = (
        json.dumps(output_schema, sort_keys=True, separators=(",", ":"))
        if output_schema is not None
        else ""
    )
    parts = [
        rendered,
        model,
        str(temperature),
        system_prompt or "",
        output_type,
        schema_fp,
    ]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


async def execute_prompt_cell(
    session: NotebookSession,
    cell_id: str,
    source: str,
    llm_config: LlmConfig,
    *,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Execute a prompt cell and return the result.

    Returns a dict compatible with CellExecutionResult fields:
        success, outputs, stdout, stderr, error, cache_hit,
        duration_ms, execution_method, artifact_uri, mutation_warnings
    """
    start_time = time.time()
    analysis = analyze_prompt_cell(source)
    output_name = analysis.name

    # Resolve model config from annotations (override llm_config defaults)
    model = analysis.model or llm_config.model
    temperature = analysis.temperature if analysis.temperature is not None else 0.0
    max_tokens = analysis.max_tokens or llm_config.max_output_tokens
    output_schema = analysis.output_schema
    # A schema implies JSON — let users omit ``@output json`` when they
    # supply ``@output_schema``.
    output_type = analysis.output_type or ("json" if output_schema is not None else "text")
    system_prompt = analysis.system_prompt

    # Load upstream variables from artifacts
    variables = _load_upstream_variables(session, cell_id)

    # Render template
    rendered = render_prompt_template(
        analysis.template_body,
        variables,
        max_tokens_per_var=2000,
    )

    if not rendered.strip():
        return _error_result("Prompt template is empty after rendering", start_time)

    provenance_hash = compute_prompt_provenance_hash(
        rendered=rendered,
        model=model,
        temperature=temperature,
        system_prompt=system_prompt,
        output_type=output_type,
        output_schema=output_schema,
    )

    # Cache check
    artifact_mgr = session.get_artifact_manager()
    notebook_id = session.notebook_state.id
    canonical_id = f"nb_{notebook_id}_cell_{cell_id}_var_{output_name}"
    var_provenance = derive_subkey(provenance_hash, output_name)

    if use_cache:
        cached = artifact_mgr.find_cached(var_provenance)
        if cached is not None:
            canonical = artifact_mgr.artifact_store.get_latest_version(canonical_id)
            if canonical is not None and canonical.provenance_hash == var_provenance:
                # Cache hit
                duration_ms = (time.time() - start_time) * 1000
                blob = artifact_mgr.load_artifact_data(canonical.id, canonical.version)
                content_type = "json/object"
                try:
                    spec = json.loads(canonical.transform_spec or "{}")
                    content_type = spec.get("params", {}).get("content_type", content_type)
                except Exception:
                    pass

                value = _parse_output(blob, content_type)
                uri = f"strata://artifact/{canonical.id}@v={canonical.version}"

                preview = _preview(value)
                display_entry = {
                    "preview": preview,
                    "content_type": content_type,
                    "bytes": len(blob),
                }
                display_text = str(preview) if not isinstance(preview, str) else preview
                display_output = {
                    "content_type": "text/markdown" if "\n" in display_text else "json/object",
                    "preview": display_text,
                    "markdown_text": display_text if "\n" in display_text else None,
                }

                return {
                    "success": True,
                    "outputs": {output_name: display_entry, "_": display_entry},
                    "display_outputs": [display_output],
                    "display_output": display_output,
                    "stdout": "",
                    "stderr": "",
                    "error": None,
                    "cache_hit": True,
                    "duration_ms": int(duration_ms),
                    "execution_method": "cached",
                    "artifact_uri": uri,
                    "mutation_warnings": [],
                }

    # Build messages
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": rendered})

    # Estimate tokens
    input_tokens_est = estimate_tokens(rendered)
    if system_prompt:
        input_tokens_est += estimate_tokens(system_prompt)

    logger.info(
        "prompt_cell_execute %s: model=%s temp=%s est_tokens=%d output_type=%s",
        cell_id,
        model,
        temperature,
        input_tokens_est,
        output_type,
    )

    # Call LLM — when a schema is set, run a validate-and-retry loop so
    # non-OpenAI providers (which only guarantee syntactic JSON) still
    # produce schema-conforming output. The retry feeds the previous
    # response and the validator errors back to the model.
    from strata.notebook.llm import LlmConfig as _LlmConfig

    call_config = _LlmConfig(
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
        model=model,
        max_output_tokens=max_tokens,
        timeout_seconds=llm_config.timeout_seconds,
    )
    max_attempts = (
        analysis.validate_retries
        if analysis.validate_retries is not None
        else DEFAULT_VALIDATE_ATTEMPTS
    )
    if output_schema is None:
        max_attempts = 1  # Nothing to validate against.

    result = None
    validation_errors: list[str] = []
    validation_retries = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for attempt in range(1, max_attempts + 1):
        try:
            result = await chat_completion(
                call_config,
                messages,
                temperature=temperature,
                output_type=output_type,
                output_schema=output_schema,
            )
        except Exception as e:
            return _error_result(f"LLM call failed: {e}", start_time)

        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens

        if output_schema is None:
            validation_errors = []
            break

        validation_errors = _validation_errors(result.content, output_schema)
        if not validation_errors:
            break

        logger.info(
            "prompt_cell_validate %s: attempt=%d errors=%d first=%r",
            cell_id,
            attempt,
            len(validation_errors),
            validation_errors[0],
        )

        if attempt < max_attempts:
            validation_retries += 1
            # Seed the retry with the model's bad answer + the validator
            # diagnostics. Sending the previous response as an assistant
            # turn (instead of pasting it into the user turn) lets the
            # model treat it as history to correct, not as the user's
            # instructions.
            messages = messages + [
                {"role": "assistant", "content": result.content},
                {"role": "user", "content": _format_retry_prompt(validation_errors)},
            ]

    if output_schema is not None and validation_errors:
        joined = "; ".join(validation_errors[:3])
        return _error_result(
            f"Response failed schema validation after {max_attempts} attempt(s): {joined}",
            start_time,
        )

    assert result is not None  # loop runs at least once

    # Parse output
    content = result.content
    if output_type == "json":
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            import re

            m = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n```", content)
            if m:
                try:
                    content = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass  # Keep as string

    # Serialize and store artifact
    content_type = "json/object"
    blob = json.dumps(content, indent=2, default=str).encode()

    try:
        from strata.artifact_store import TransformSpec

        version = artifact_mgr.artifact_store.create_artifact(
            artifact_id=canonical_id,
            provenance_hash=var_provenance,
            transform_spec=TransformSpec(
                executor="prompt",
                params={
                    "content_type": content_type,
                    "model": model,
                    "temperature": temperature,
                    "output_type": output_type,
                    # Totals across the validate-and-retry loop, so cost
                    # accounting reflects what was actually spent.
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "validation_retries": validation_retries,
                    **({"output_schema": output_schema} if output_schema is not None else {}),
                },
                inputs=[],
            ),
        )
        artifact_mgr.artifact_store.write_blob(canonical_id, version, blob)
        artifact_mgr.artifact_store.finalize_artifact(
            canonical_id,
            version,
            schema_json="{}",
            row_count=1,
            byte_size=len(blob),
        )
        artifact_uri = f"strata://artifact/{canonical_id}@v={version}"
    except Exception as e:
        logger.error("Failed to store prompt cell artifact: %s", e)
        artifact_uri = None

    duration_ms = (time.time() - start_time) * 1000

    # Build display output for the frontend
    preview = _preview(content)
    display_entry = {
        "preview": preview,
        "content_type": content_type,
        "bytes": len(blob),
    }

    # The '_' key is the display value the frontend renders
    outputs = {
        output_name: display_entry,
        "_": display_entry,
    }

    # Build a display_output dict the frontend can render as text/markdown
    display_text = str(preview) if not isinstance(preview, str) else preview
    display_output = {
        "content_type": "text/markdown" if "\n" in display_text else "json/object",
        "scalar": display_text,
        "markdown_text": display_text if "\n" in display_text else None,
    }

    retries_suffix = f" | Validation retries: {validation_retries}" if validation_retries else ""
    return {
        "success": True,
        "outputs": outputs,
        "display_outputs": [display_output],
        "display_output": display_output,
        "stdout": "",
        "stderr": (
            f"Model: {result.model} | "
            f"Tokens: {total_input_tokens}→{total_output_tokens}{retries_suffix}"
        ),
        "error": None,
        "cache_hit": False,
        "duration_ms": int(duration_ms),
        "execution_method": "llm",
        "artifact_uri": artifact_uri,
        "mutation_warnings": [],
        "validation_retries": validation_retries,
    }


def _load_upstream_variables(
    session: NotebookSession,
    cell_id: str,
) -> dict[str, Any]:
    """Load upstream variable values from artifacts."""
    variables: dict[str, Any] = {}
    cell = session.notebook_state.get_cell(cell_id)
    if cell is None:
        return variables

    artifact_mgr = session.get_artifact_manager()
    notebook_id = session.notebook_state.id

    for upstream_id in cell.upstream_ids:
        upstream_cell = session.notebook_state.get_cell(upstream_id)
        if upstream_cell is None:
            continue

        referenced_vars = [v for v in cell.references if v in upstream_cell.defines]

        for var_name in referenced_vars:
            canonical_id = f"nb_{notebook_id}_cell_{upstream_id}_var_{var_name}"
            try:
                artifact = artifact_mgr.artifact_store.get_latest_version(canonical_id)
                if artifact is None:
                    continue
                blob = artifact_mgr.load_artifact_data(canonical_id, artifact.version)
                content_type = "json/object"
                if artifact.transform_spec:
                    try:
                        spec = json.loads(artifact.transform_spec)
                        ct = spec.get("params", {}).get("content_type")
                        if ct:
                            content_type = ct
                    except (ValueError, KeyError):
                        pass
                variables[var_name] = _parse_output(blob, content_type)
            except Exception as e:
                logger.warning("Failed to load upstream %s: %s", var_name, e)

    return variables


def _parse_output(blob: bytes, content_type: str) -> Any:
    """Parse artifact blob back to a Python value."""
    if content_type == "arrow/ipc":
        try:
            import pyarrow as pa

            reader = pa.ipc.open_stream(blob)
            table = reader.read_all().combine_chunks()
            try:
                return table.to_pandas()
            except Exception:
                return table
        except Exception:
            return blob.decode(errors="replace")
    elif content_type == "json/object":
        try:
            return json.loads(blob)
        except Exception:
            return blob.decode(errors="replace")
    elif content_type == "pickle/object":
        import pickle

        try:
            return pickle.loads(blob)  # noqa: S301
        except Exception:
            return blob.decode(errors="replace")
    return blob.decode(errors="replace")


def _preview(value: Any) -> Any:
    """Create a JSON-safe preview of a value."""
    if isinstance(value, str):
        return value[:4000] if len(value) > 4000 else value
    if isinstance(value, (int, float, bool, type(None))):
        return value
    if isinstance(value, (dict, list)):
        text = json.dumps(value, indent=2, default=str)
        return text[:500] if len(text) > 500 else text
    return str(value)[:4000]


def _error_result(error: str, start_time: float) -> dict[str, Any]:
    """Build an error result dict."""
    return {
        "success": False,
        "outputs": {},
        "stdout": "",
        "stderr": "",
        "error": error,
        "cache_hit": False,
        "duration_ms": int((time.time() - start_time) * 1000),
        "execution_method": "llm",
        "artifact_uri": None,
        "mutation_warnings": [],
    }
