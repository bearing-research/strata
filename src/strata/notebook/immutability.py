"""Runtime mutation detection for notebook cell inputs.

This module provides heuristic, best-effort mutation detection for input variables.
It's conservative: if we can't prove a variable wasn't mutated, we report a warning.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any, TypedDict


@dataclass
class InputSnapshot:
    """Snapshot of an input variable for mutation detection."""

    var_name: str
    identity: int  # id(obj) at snapshot time
    content_hash: str | None  # sample-based hash for DataFrames


class MutationWarning(TypedDict):
    """Warning about a detected mutation.

    TypedDict rather than a dataclass because the only consumers
    are JSON writers (manifest.json, WS cell_output payloads), so
    keeping the wire format identical to the detection format
    eliminates a round of serialization.
    """

    var_name: str
    message: str
    suggestion: str | None


def snapshot_inputs(namespace: dict[str, Any], input_names: list[str]) -> list[InputSnapshot]:
    """Take snapshots of input variables before cell execution.

    Captures ``id(value)`` for every input, plus a sample content hash for
    pandas DataFrames / Series (the only types this module can detect in-place
    mutation for).

    Parameters
    ----------
    namespace : dict
        The namespace dict containing variables.
    input_names : list of str
        Input variable names to snapshot.

    Returns
    -------
    list of InputSnapshot
        One snapshot per input present in *namespace*.
    """
    snapshots = []

    for var_name in input_names:
        if var_name not in namespace:
            continue

        value = namespace[var_name]
        var_id = id(value)

        # For DataFrames, compute a sample hash
        content_hash = None
        try:
            import pandas as pd

            if isinstance(value, (pd.DataFrame, pd.Series)):
                content_hash = _hash_dataframe_sample(value)
        except ImportError:
            pass

        snapshots.append(
            InputSnapshot(
                var_name=var_name,
                identity=var_id,
                content_hash=content_hash,
            )
        )

    return snapshots


def detect_mutations(
    namespace: dict[str, Any], snapshots: list[InputSnapshot]
) -> list[MutationWarning]:
    """Detect mutations by comparing current state against snapshots.

    Detection is best-effort and limited to what :func:`snapshot_inputs`
    captured:

    - **Deletion** — the variable is gone from *namespace*.
    - **Reassignment** — ``id(current) != id(original)``; treated as *not* a
      mutation (the input object itself was untouched), so it is skipped.
    - **In-place DataFrame/Series mutation** — same identity but a different
      sample hash.

    Other types (dict, list, opaque objects) carry no snapshot state, so their
    in-place mutations are not detected.

    Parameters
    ----------
    namespace : dict
        The namespace dict after execution.
    snapshots : list of InputSnapshot
        Snapshots taken before execution.

    Returns
    -------
    list of MutationWarning
        One warning per detected mutation (empty if none).
    """
    warnings = []

    for snapshot in snapshots:
        if snapshot.var_name not in namespace:
            # Variable was deleted — report as mutation
            warnings.append(
                MutationWarning(
                    var_name=snapshot.var_name,
                    message=f"'{snapshot.var_name}' was deleted during execution",
                    suggestion=None,
                )
            )
            continue

        current_value = namespace[snapshot.var_name]
        current_id = id(current_value)

        # If identity changed, it was reassigned (not a mutation)
        if current_id != snapshot.identity:
            continue

        # Same identity — check if the object was mutated
        mutation_detected = _check_object_mutation(current_value, snapshot)

        if mutation_detected:
            message, suggestion = mutation_detected
            warnings.append(
                MutationWarning(
                    var_name=snapshot.var_name,
                    message=message,
                    suggestion=suggestion,
                )
            )

    return warnings


def _check_object_mutation(value: Any, snapshot: InputSnapshot) -> tuple[str, str | None] | None:
    """Check whether a same-identity object was mutated in place.

    Only pandas DataFrame / Series carry a snapshot hash to compare against;
    every other type returns ``None`` (no detectable mutation).

    Parameters
    ----------
    value : Any
        The current value (same ``id`` as at snapshot time).
    snapshot : InputSnapshot
        The pre-execution snapshot.

    Returns
    -------
    tuple of (str, str or None), or None
        ``(message, suggestion)`` when a mutation is detected, else ``None``.
    """
    try:
        import pandas as pd
    except ImportError:
        return None

    if isinstance(value, (pd.DataFrame, pd.Series)) and snapshot.content_hash:
        current_hash = _hash_dataframe_sample(value)
        if current_hash != snapshot.content_hash:
            return (
                f"'{snapshot.var_name}' was mutated without reassignment",
                "Consider using df = df.copy() or df = df.drop(...) instead of inplace=True",
            )
    return None


def _hash_dataframe_sample(df: Any) -> str | None:
    """Hash the first 5 + last 5 rows of a DataFrame for mutation detection.

    A sample-based hash that avoids full-table hashing — fast for any size and
    catches most in-place edits. Returns ``None`` when the sample can't be
    JSON-encoded (e.g. object columns holding unserializable values), which
    callers treat as "no hash captured" and skip detection for.

    Parameters
    ----------
    df : pandas.DataFrame or pandas.Series
        The value to sample-hash.

    Returns
    -------
    str or None
        Hex digest, or ``None`` if the sample couldn't be hashed.
    """
    h = hashlib.sha256()

    h.update(str(df.shape).encode())
    try:
        h.update(str(df.dtypes.to_dict()).encode())
    except AttributeError:
        # Series have a single .dtype, not .dtypes.to_dict().
        h.update(str(df.dtype).encode())

    try:
        h.update(df.head(5).to_json().encode())
        if len(df) > 5:
            h.update(df.tail(5).to_json().encode())
    except (ValueError, OverflowError, TypeError):
        # to_json() chokes on some object-dtype payloads; degrade gracefully.
        return None

    return h.hexdigest()


def apply_defensive_copy(value: Any, content_type: str) -> Any:
    """Return a defensive copy of an input value, chosen by content-type tier.

    Not currently wired into the execution path (inputs are re-read from the
    artifact store each run); kept as the building block for opt-in input
    isolation. Content-type strings are the literal serializer wire values —
    this module is loaded inside the notebook venv and can't import
    ``ContentType``.

    Tiers
    -----
    - ``arrow/ipc`` — no copy (deserialization already yields a fresh object).
    - ``json/object`` — shallow ``copy.copy``.
    - ``pickle/object`` — ``copy.deepcopy`` (safer for nested structures).

    Parameters
    ----------
    value : Any
        The value to copy.
    content_type : str
        The content type from the input spec.

    Returns
    -------
    Any
        A defensive copy, or the original when no copy is needed.
    """
    if content_type == "json/object":
        return copy.copy(value)
    if content_type == "pickle/object":
        return copy.deepcopy(value)
    # arrow/ipc (fresh on deserialize) or unknown — return as-is.
    return value
