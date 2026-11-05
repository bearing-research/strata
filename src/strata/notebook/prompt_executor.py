"""Executor for prompt-type notebook cells.

Resolves upstream artifacts to text, renders the prompt template,
calls the LLM, and stores the response as an artifact.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vestigo.provenance import derive_subkey

from strata.notebook.annotations import parse_annotations
from strata.notebook.llm import (
    LlmCompletionResult,
    LlmConfig,
    chat_completion,
    chat_completion_stream,
    estimate_tokens,
    infer_provider_name,
    render_prompt_template,
)
from strata.notebook.prompt_analyzer import analyze_prompt_cell
from strata.notebook.prompt_template import Placeholder, PromptTemplateError, parse_template
from strata.notebook.serializer import _META_SHAPE, _SHAPE_SCALAR, _extract_scalar_from_table

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from strata.notebook.session import NotebookSession

logger = logging.getLogger(__name__)

# Every attempt costs tokens, so keep the ceiling low; ``# @validate_retries N`` raises it.
DEFAULT_VALIDATE_ATTEMPTS = 3
_MAX_TOKENS_PER_VARIABLE = 2000


def _validation_errors(content: str, schema: dict[str, Any]) -> list[str]:
    """Return path-addressed validation errors; empty means valid JSON matching the schema.

    A JSON parse failure comes back as a single error in the same list.
    """
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return [f"response is not valid JSON: {exc.msg} (line {exc.lineno}, col {exc.colno})"]

    try:
        import jsonschema
    except ImportError:
        # No validator available: fall back to "is it JSON".
        return []

    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(parsed), key=lambda e: list(e.absolute_path))
    if not errors:
        return []
    return [_format_one_error(err) for err in errors]


def _coerce_json_text(content: str) -> str:
    """Best-effort extraction of a JSON document from model output.

    Unwraps code fences or prose around JSON. Returns the original content when
    nothing parses, so its parse error drives the retry feedback.
    """
    try:
        json.loads(content)
        return content
    except json.JSONDecodeError:
        pass

    import re

    fence = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n```", content)
    if fence:
        candidate = fence.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = content.find(opener)
        end = content.rfind(closer)
        if 0 <= start < end:
            candidate = content[start : end + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
    return content


def _format_one_error(err: Any) -> str:
    """Render a jsonschema ValidationError as ``path: message`` with a JSON Pointer path."""
    pointer = "/" + "/".join(str(part) for part in err.absolute_path) if err.absolute_path else ""
    return f"{pointer or '(root)'}: {err.message}"


def _format_retry_prompt(errors: list[str]) -> str:
    """Build the user-turn feedback message for a validation retry.

    Kept spare: extra framing burns tokens and can pull the model away from the schema.
    """
    bullets = "\n".join(f"- {err}" for err in errors)
    return (
        "Your previous response did not validate against the required schema.\n"
        f"Errors:\n{bullets}\n"
        "Return a corrected JSON object that satisfies the schema."
    )


def prompt_reopen_identity(cell: Any, session: Any) -> str | None:
    """What a reopened prompt cell's cached answer depends on beyond its text.

    Model, sampling, length and output shape are not in the generic provenance
    (source, env, inputs), so without this a changed ``[ai]`` model would show the
    old answer as ready. Never ``None``.
    """
    from strata.notebook.llm.config import read_notebook_ai_config

    analysis = analyze_prompt_cell(cell.source)
    # The notebook's own [ai] block, not the resolved config: resolving needs an API
    # key, so it depends on the caller's context and the identity would never match.
    notebook_ai = read_notebook_ai_config(session) or {}
    payload = {
        "model": analysis.model or notebook_ai.get("model"),
        "temperature": analysis.temperature if analysis.temperature is not None else 0.0,
        "max_tokens": analysis.max_tokens or notebook_ai.get("max_output_tokens"),
        "system_prompt": analysis.system_prompt,
        "output_type": analysis.output_type
        or ("json" if analysis.output_schema is not None else "text"),
        "output_schema": analysis.output_schema,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def compute_prompt_provenance_hash(
    *,
    rendered: str,
    model: str,
    temperature: float,
    system_prompt: str | None,
    output_type: str,
    output_schema: dict[str, Any] | None,
    max_tokens: int | None = None,
    input_hashes: dict[str, str] | None = None,
) -> str:
    """Stable cache key for a prompt-cell invocation.

    Folds in the schema fingerprint and ``max_tokens`` (both change the answer
    asked for) and each upstream's provenance hash: rendering truncates variables
    at ``max_tokens_per_var``, so the rendered template alone misses edits past
    the cut.
    """
    schema_fp = (
        json.dumps(output_schema, sort_keys=True, separators=(",", ":"))
        if output_schema is not None
        else ""
    )
    inputs_fp = json.dumps(sorted((input_hashes or {}).items()), separators=(",", ":"))
    parts = [
        rendered,
        model,
        str(temperature),
        system_prompt or "",
        output_type,
        schema_fp,
        str(max_tokens) if max_tokens is not None else "",
        inputs_fp,
    ]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


async def execute_prompt_cell(
    session: NotebookSession,
    cell_id: str,
    source: str,
    llm_config: LlmConfig,
    *,
    use_cache: bool = True,
    on_delta: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Execute a prompt cell and return a dict of ``CellExecutionResult`` fields.

    ``on_delta`` receives one ``CELL_OUTPUT_DELTA`` payload
    (``{cell_id, attempt, kind, text}``) per streaming event. Cache hits fire no
    deltas; callback failures are logged and never fail the cell.
    """
    start_time = time.time()
    analysis = analyze_prompt_cell(source)
    output_name = analysis.name
    if parse_annotations(source).nocache:
        use_cache = False

    model = analysis.model or llm_config.model
    temperature = analysis.temperature if analysis.temperature is not None else 0.0
    max_tokens = analysis.max_tokens or llm_config.max_output_tokens
    output_schema = analysis.output_schema
    # A schema implies JSON, so ``@output json`` is optional with ``@output_schema``.
    output_type = analysis.output_type or ("json" if output_schema is not None else "text")
    system_prompt = analysis.system_prompt

    variables, input_hashes = _load_upstream_variables(session, cell_id)
    object_names = {name for name, value in variables.items() if isinstance(value, ObjectInput)}
    object_problems = _reaches_into_objects(analysis.template_body, object_names)
    object_problems += await _objects_to_text(session, variables)
    if object_problems:
        return _error_result("\n".join(object_problems), start_time)

    try:
        rendered = render_prompt_template(
            analysis.template_body,
            variables,
            max_tokens_per_var=_MAX_TOKENS_PER_VARIABLE,
        )
    except PromptTemplateError as exc:
        return _error_result(str(exc), start_time)

    if not rendered.strip():
        return _error_result("Prompt template is empty after rendering", start_time)

    provenance_hash = compute_prompt_provenance_hash(
        rendered=rendered,
        model=model,
        temperature=temperature,
        system_prompt=system_prompt,
        output_type=output_type,
        output_schema=output_schema,
        max_tokens=max_tokens,
        input_hashes=input_hashes,
    )

    artifact_mgr = session.get_artifact_manager()
    notebook_id = session.notebook_state.id
    canonical_id = f"nb_{notebook_id}_cell_{cell_id}_var_{output_name}"
    var_provenance = derive_subkey(provenance_hash, output_name)

    if use_cache:
        cached = artifact_mgr.find_cached(var_provenance)
        if cached is not None:
            canonical = artifact_mgr.artifact_store.get_latest_version(canonical_id)
            if canonical is not None and canonical.provenance_hash == var_provenance:
                duration_ms = (time.time() - start_time) * 1000
                blob = artifact_mgr.load_artifact_data(canonical.id, canonical.version)
                content_type = canonical.params.get("content_type", "json/object")

                value = _parse_output(blob, content_type)
                uri = f"strata://artifact/{canonical.id}@v={canonical.version}"
                _publish(session, cell_id, output_name, uri)

                preview = _preview(value)
                display_entry = {
                    "preview": preview,
                    "content_type": content_type,
                    "bytes": len(blob),
                }
                display_text = str(preview) if not isinstance(preview, str) else preview
                # Same key as the fresh-run path: the frontend renders ``output.scalar``.
                display_output = {
                    "content_type": "text/markdown" if "\n" in display_text else "json/object",
                    "scalar": display_text,
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

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": rendered})

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

    # With a schema, validate and retry so providers that only guarantee syntactic
    # JSON still conform. ``replace``, not a new config: the base_url's guard travels with it.
    call_config = dataclasses.replace(llm_config, model=model, max_output_tokens=max_tokens)
    max_attempts = (
        analysis.validate_retries
        if analysis.validate_retries is not None
        else DEFAULT_VALIDATE_ATTEMPTS
    )
    if output_schema is None:
        max_attempts = 1  # Nothing to validate against.

    # Anthropic + schema uses native tool-use, which has no streaming path yet, so
    # it stays unary and emits no deltas.
    use_streaming = on_delta is not None and not (
        output_schema is not None and infer_provider_name(call_config.base_url) == "anthropic"
    )

    result = None
    validation_errors: list[str] = []
    validation_retries = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for attempt in range(1, max_attempts + 1):
        try:
            if use_streaming:
                result = await _stream_completion(
                    call_config,
                    messages,
                    cell_id=cell_id,
                    attempt=attempt,
                    temperature=temperature,
                    output_type=output_type,
                    output_schema=output_schema,
                    on_delta=on_delta,
                )
            else:
                result = await chat_completion(
                    call_config,
                    messages,
                    temperature=temperature,
                    output_type=output_type,
                    output_schema=output_schema,
                )
        except Exception as e:
            # ``str()`` of an httpx timeout is empty; its type still says what happened.
            return _error_result(f"LLM call failed: {str(e) or type(e).__name__}", start_time)

        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens

        if output_schema is None:
            validation_errors = []
            break

        # Providers without enforcement often fence or preface JSON; extract it instead
        # of burning a retry.
        coerced = _coerce_json_text(result.content)
        validation_errors = _validation_errors(coerced, output_schema)
        if not validation_errors:
            result.content = coerced
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
            # Clear the frontend stream buffer so attempt N's invalid JSON doesn't fuse with
            # N+1's output. ``text`` carries the first validator error as a retry notice.
            await _emit_delta(
                on_delta,
                {
                    "cell_id": cell_id,
                    "attempt": attempt + 1,
                    "kind": "retry",
                    "text": validation_errors[0],
                },
            )
            # The bad answer goes in as an assistant turn so the model treats it as history
            # to correct, not as user instructions.
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

    content = result.content
    if output_type == "json":
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            # JSON in a markdown code block
            import re

            m = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n```", content)
            if m:
                try:
                    content = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass  # Keep as string

    content_type = "json/object"
    blob = json.dumps(content, indent=2, default=str).encode()

    try:
        from vestigo.provenance import TransformSpec
        from vestigo.store import StagedVersion

        from strata.notebook.artifact_integration import caller_principal_id

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
                    # Totals across retries, so cost accounting reflects what was spent.
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "validation_retries": validation_retries,
                    **({"output_schema": output_schema} if output_schema is not None else {}),
                },
                inputs=[],
            ),
            principal=caller_principal_id(),
        )
        artifact_mgr.artifact_store.write_blob(canonical_id, version, blob)
        # Canonical, like every other cell output: a duplicated cell reads by its own id.
        artifact_mgr.finalize_cell_outputs(
            [
                StagedVersion(
                    artifact_id=canonical_id,
                    version=version,
                    schema_json="{}",
                    row_count=1,
                    byte_size=len(blob),
                    content_sha256=hashlib.sha256(blob).hexdigest(),
                )
            ]
        )
        artifact_uri = f"strata://artifact/{canonical_id}@v={version}"
        _publish(session, cell_id, output_name, artifact_uri)
    except Exception as e:
        logger.error("Failed to store prompt cell artifact: %s", e)
        artifact_uri = None

    duration_ms = (time.time() - start_time) * 1000

    preview = _preview(content)
    display_entry = {
        "preview": preview,
        "content_type": content_type,
        "bytes": len(blob),
    }

    # '_' is the display value the frontend renders
    outputs = {
        output_name: display_entry,
        "_": display_entry,
    }

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


def _publish(session: NotebookSession, cell_id: str, output_name: str, uri: str) -> None:
    """Put the answer on the cell, as other cells' outputs are, before anything downstream
    is keyed: a cell run next in the same run reads its upstreams from there."""
    cell = session.notebook_state.get_cell(cell_id)
    if cell is not None:
        cell.artifact_uris[output_name] = uri
        cell.artifact_uri = uri


async def _emit_delta(
    on_delta: Callable[[dict[str, Any]], Awaitable[None]] | None,
    payload: dict[str, Any],
) -> None:
    """Fire the streaming callback, swallowing failures.

    A client gone mid-stream must not fail the cell; the result still reaches
    the store and the final ``cell_output`` frame.
    """
    if on_delta is None:
        return
    try:
        await on_delta(payload)
    except Exception:
        logger.warning(
            "on_delta callback failed for cell %s", payload.get("cell_id"), exc_info=True
        )


async def _stream_completion(
    config: LlmConfig,
    messages: list[dict[str, str]],
    *,
    cell_id: str,
    attempt: int,
    temperature: float | None,
    output_type: str | None,
    output_schema: dict[str, Any] | None,
    on_delta: Callable[[dict[str, Any]], Awaitable[None]] | None,
) -> LlmCompletionResult:
    """Stream one completion attempt, forwarding deltas as they arrive.

    Returns the same ``LlmCompletionResult`` shape as the unary path; usage
    comes from the stream's final ``done`` event.
    """
    content_parts: list[str] = []
    model = config.model
    input_tokens = 0
    output_tokens = 0
    degraded = False

    async for event in chat_completion_stream(
        config,
        messages,
        temperature=temperature,
        output_type=output_type,
        output_schema=output_schema,
    ):
        if event["type"] == "delta":
            content_parts.append(event["text"])
            await _emit_delta(
                on_delta,
                {
                    "cell_id": cell_id,
                    "attempt": attempt,
                    "kind": "delta",
                    "text": event["text"],
                },
            )
        elif event["type"] == "notice":
            # Provider degradation notice (e.g. response_format rejected): surface it on the
            # stream without polluting the accumulated content.
            logger.info("prompt_cell_notice %s: %s", cell_id, event["text"])
            await _emit_delta(
                on_delta,
                {
                    "cell_id": cell_id,
                    "attempt": attempt,
                    "kind": "notice",
                    "text": event["text"],
                },
            )
        elif event["type"] == "done":
            model = event.get("model", model)
            input_tokens = event.get("input_tokens", 0)
            output_tokens = event.get("output_tokens", 0)
            degraded = bool(event.get("degraded", False))

    return LlmCompletionResult(
        content="".join(content_parts),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        degraded=degraded,
    )


def _load_upstream_variables(
    session: NotebookSession,
    cell_id: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load upstream values and their provenance hashes as ``(variables, input_hashes)``.

    The hashes key the cache because rendering truncates long variables.
    """
    variables: dict[str, Any] = {}
    input_hashes: dict[str, str] = {}
    cell = session.notebook_state.get_cell(cell_id)
    if cell is None:
        return variables, input_hashes

    artifact_mgr = session.get_artifact_manager()
    notebook_id = session.notebook_state.id

    for upstream_id in cell.upstream_ids:
        upstream_cell = session.notebook_state.get_cell(upstream_id)
        if upstream_cell is None:
            continue

        wired = session.wired_variables(cell_id, upstream_id)
        referenced_vars = [v for v in cell.references if v in wired]

        for var_name in referenced_vars:
            canonical_id = f"nb_{notebook_id}_cell_{upstream_id}_var_{var_name}"
            try:
                artifact = artifact_mgr.artifact_store.get_latest_version(canonical_id)
                if artifact is None:
                    continue
                blob = artifact_mgr.load_artifact_data(canonical_id, artifact.version)
                content_type = artifact.params.get("content_type") or "json/object"
                value = _parse_output(blob, content_type)
                if isinstance(value, ObjectInput):
                    value = dataclasses.replace(value, ref=f"{canonical_id}@v={artifact.version}")
                variables[var_name] = value
                input_hashes[var_name] = artifact.provenance_hash
            except Exception as e:
                logger.warning("Failed to load upstream %s: %s", var_name, e)

    return variables, input_hashes


@dataclasses.dataclass(frozen=True, repr=False)
class ObjectInput:
    """A stored value the server must not load: only the notebook's venv reads it."""

    content_type: str
    blob: bytes
    # The artifact version it was read from, which never changes, so its text does not.
    ref: str | None = None


_LOADABLE_OBJECT_TYPES = frozenset({"pickle/object", "module/cell-instance"})
_OBJECT_TEXT_HARNESS = Path(__file__).parent / "object_text_harness.py"
_OBJECT_TEXT_TIMEOUT_SECONDS = 30.0
# What ``object.__repr__`` and functions print. Its address changes on every load, so
# the rendered prompt would too, and no run would hit the cache.
_DEFAULT_REPR_RE = re.compile(r"<[^<>]* at 0x[0-9a-fA-F]+>")
# Each object's text by artifact version: a cache hit must not load a large model again.
_OBJECT_TEXTS: OrderedDict[str, str] = OrderedDict()
_OBJECT_TEXTS_MAX = 256


async def _objects_to_text(session: NotebookSession, variables: dict[str, Any]) -> list[str]:
    """Replace each ``ObjectInput`` with its text; return why any could not be."""
    names = [name for name, value in variables.items() if isinstance(value, ObjectInput)]
    results = await asyncio.gather(*(_object_text(session, n, variables[n]) for n in names))
    problems: list[str] = []
    for name, (text, problem) in zip(names, results, strict=True):
        if problem is not None:
            problems.append(problem)
        else:
            variables[name] = text
    return problems


def _reaches_into_objects(template: str, object_names: set[str]) -> list[str]:
    """``{{ obj.attr }}`` on an object: its attributes are not in the text it was read as."""
    parts, _problems = parse_template(template)
    return [
        f"`{part.text}`: `{part.root}` is a Python object, which a prompt reads only as "
        f"its text. Take what the prompt needs from it in the cell that produces it."
        for part in parts
        if isinstance(part, Placeholder)
        and part.root in object_names
        and part.expression != part.root
    ]


async def _object_text(
    session: NotebookSession, name: str, value: ObjectInput
) -> tuple[str, str | None]:
    """``str()`` of a stored object, from a subprocess running as cell code."""
    from strata.notebook.executor import _no_interpreter_message
    from strata.notebook.harness_env import configured_allowlist, harness_env
    from strata.notebook.harness_user import (
        LocalExecutionRefused,
        hand_over,
        identity_env,
        resolve_harness_user,
        spawn_kwargs,
    )
    from strata.notebook.process_tree import (
        subprocess_kwargs_for_new_group,
        terminate_subprocess_tree,
    )

    convert = f"Convert `{name}` to text, a dict or a table in the cell that produces it."
    if value.ref is not None and value.ref in _OBJECT_TEXTS:
        _OBJECT_TEXTS.move_to_end(value.ref)
        return _OBJECT_TEXTS[value.ref], None
    if value.content_type not in _LOADABLE_OBJECT_TYPES:
        return "", f"`{name}` is stored as {value.content_type}, which has no text. {convert}"
    try:
        harness_user = resolve_harness_user()
    except LocalExecutionRefused as exc:
        return "", f"`{name}` is a Python object, and {exc} {convert}"
    python_executable = session.venv_python
    if python_executable is None:
        return "", f"`{name}` is a Python object: {_no_interpreter_message(session)}"

    with tempfile.TemporaryDirectory(prefix="strata_prompt_object_") as directory:
        path = Path(directory) / "input.pickle"
        path.write_bytes(value.blob)
        hand_over(Path(directory), harness_user)
        allowlist = configured_allowlist()
        # A fixed seed: a set prints in one order, so the rendered prompt and its cache
        # key do not change from run to run.
        base = harness_env(allowlist) if allowlist else dict(os.environ)
        env = {**(identity_env(base, harness_user) or base), "PYTHONHASHSEED": "0"}
        try:
            process = await asyncio.create_subprocess_exec(
                str(python_executable),
                str(_OBJECT_TEXT_HARNESS),
                value.content_type,
                str(path),
                # One past the budget, so the renderer still marks the cut.
                str(_MAX_TOKENS_PER_VARIABLE * 4 + 1),
                cwd=str(session.path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                **spawn_kwargs(harness_user),
                **subprocess_kwargs_for_new_group(),
            )
        except OSError as exc:
            return "", f"`{name}` could not be read: {exc}. {convert}"
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=_OBJECT_TEXT_TIMEOUT_SECONDS
            )
        except TimeoutError:
            return "", f"`str({name})` did not finish in time. {convert}"
        finally:
            # Also on cancel: the object's code must not outlive the cell's run.
            if process.returncode is None:
                await asyncio.shield(terminate_subprocess_tree(process))

    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        detail = stderr.decode(errors="replace").strip()[-500:]
        return "", f"`{name}` could not be read: {detail or 'the process died'}. {convert}"
    if "error" in result:
        return "", f"`{name}` could not be read: {result['error']}. {convert}"
    text, type_name = result["text"], result["type"]
    if _DEFAULT_REPR_RE.search(text):
        return "", (
            f"`{name}` ({type_name}) has no text of its own: `str({name})` gives `{text[:200]}`. "
            "Give its class a `__str__`, or convert it in the cell that produces it."
        )
    if value.ref is not None:
        _OBJECT_TEXTS[value.ref] = text
        if len(_OBJECT_TEXTS) > _OBJECT_TEXTS_MAX:
            _OBJECT_TEXTS.popitem(last=False)
    return text, None


def _parse_output(blob: bytes, content_type: str) -> Any:
    """Parse artifact blob back to a Python value."""
    if content_type == "arrow/ipc":
        try:
            import pyarrow as pa

            reader = pa.ipc.open_stream(blob)
            table = reader.read_all().combine_chunks()
            # datetime / Decimal / numpy scalars are stored as one-row tables.
            if (table.schema.metadata or {}).get(_META_SHAPE) == _SHAPE_SCALAR:
                return _extract_scalar_from_table(table)
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
    elif content_type.startswith("text/"):
        return blob.decode(errors="replace")
    # Unpickling would run the producing cell's code as the server user.
    return ObjectInput(content_type, blob)


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
