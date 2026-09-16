"""The generated TypeScript and the payload models must not drift apart.

``frontend/src/types/ws-payloads.generated.ts`` is committed so the frontend
build needs no Python step. That only helps while it matches the models it was
generated from, which is what these check: a field added to a payload model
without regenerating leaves the frontend compiling against a type that no longer
describes the wire.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from strata.notebook.protocol import MessageType
from strata.notebook.ws_payloads import FRAME_PAYLOADS, WsPayload

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "generate_ws_types.py"
_GENERATED = _REPO / "frontend" / "src" / "types" / "ws-payloads.generated.ts"


def test_the_committed_typescript_matches_the_models() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--check"],
        capture_output=True,
        text=True,
        cwd=_REPO,
    )
    # Distinguish the two ways this exits nonzero. The generator raises
    # UnsupportedSchema on a construct it cannot render, and telling that
    # developer to regenerate sends them to run the same crash again.
    crashed = "Traceback" in result.stderr
    assert result.returncode == 0, (
        f"the generator failed on a schema it cannot render:\n{result.stderr}"
        if crashed
        else (
            f"{_GENERATED.relative_to(_REPO)} is stale.\n"
            "Run: uv run python scripts/generate_ws_types.py\n"
            f"{result.stderr}"
        )
    )


def test_every_registered_frame_is_a_real_message_type() -> None:
    # A frame key that is not a MessageType would generate a payload entry the
    # server can never send.
    for frame in FRAME_PAYLOADS:
        assert isinstance(frame, MessageType)


def test_every_payload_model_is_registered() -> None:
    """A model nobody registers is invisible to the frontend.

    Typing a frame and then not listing it means the work looks done from the
    Python side while the client still gets ``unknown`` -- exactly the gap the
    registry exists to close. Nested models (the building blocks of a payload)
    are excluded: they reach the frontend through the payload that holds them.
    """
    import strata.notebook.ws_payloads as module

    registered = set(FRAME_PAYLOADS.values())
    # Reachability, not a name suffix. "endswith('Model')" happens to match
    # today's nested models, but it would fail a future nested payload named
    # otherwise, and would silently skip the guard for a frame payload that
    # happened to end in Model -- exempting exactly the case the test exists
    # to catch.
    reachable: set[str] = set()
    for model in registered:
        reachable.update(model.model_json_schema().get("$defs", {}))

    unregistered = {
        name
        for name, obj in vars(module).items()
        if isinstance(obj, type)
        and issubclass(obj, WsPayload)
        and obj is not WsPayload
        and name not in reachable
        and obj not in registered
    }
    assert not unregistered, (
        f"payload models not in FRAME_PAYLOADS: {sorted(unregistered)} — "
        "register them so the frontend gets their types"
    )


def test_a_field_too_long_for_one_line_is_broken_as_prettier_breaks_it() -> None:
    """The frontend's prettier hook rewrites the committed file; a line the
    emitter left too long would come back as drift on the next ``--check``."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("generate_ws_types", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._field_line("code", "?", "'a' | null") == "  code?: 'a' | null"
    fits_below = " | ".join(f"'code_{i}'" for i in range(9))
    assert module._field_line("code", "?", fits_below) == f"  code?:\n    {fits_below}"
    too_long = " | ".join(f"'code_number_{i}'" for i in range(8))
    assert module._field_line("code", "?", too_long) == "  code?:\n" + "\n".join(
        f"    | 'code_number_{i}'" for i in range(8)
    )
