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
    assert result.returncode == 0, (
        f"{_GENERATED.relative_to(_REPO)} is stale.\n"
        "Run: uv run python scripts/generate_ws_types.py\n"
        f"{result.stderr}"
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
    nested = {
        name
        for name, obj in vars(module).items()
        if isinstance(obj, type) and issubclass(obj, WsPayload) and name.endswith("Model")
    }
    unregistered = {
        name
        for name, obj in vars(module).items()
        if isinstance(obj, type)
        and issubclass(obj, WsPayload)
        and obj is not WsPayload
        and name not in nested
        and obj not in registered
    }
    assert not unregistered, (
        f"payload models not in FRAME_PAYLOADS: {sorted(unregistered)} — "
        "register them so the frontend gets their types"
    )
