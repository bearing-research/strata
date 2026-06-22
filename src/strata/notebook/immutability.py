"""Runtime mutation detection for notebook cell inputs.

Heuristic, best-effort detection that a cell mutated one of its inputs in place
(rather than reassigning it) — the residual cases the static analyzer can't see
(aliases, helper-function mutation, bare method mutators). Each input gets an
identity check plus a cheap, sampled content fingerprint; a same-identity value
whose fingerprint changed is reported as a mutation. Fingerprints come from an
extensible registry (pandas / numpy / mappings / sequences / sized containers);
unknown types fall back to identity-only. Detection is warn-only — see
``docs/internal/design-mutation-fingerprint-registry.md``.
"""

from __future__ import annotations

import collections.abc
import copy
import hashlib
from dataclasses import dataclass
from typing import Any, NamedTuple, TypedDict

# Sampling bounds for content fingerprints: never hash a whole value (this runs
# on every input of every cell execution), only a head/tail sample.
_MAX_SAMPLE = 5
_MAX_REPR = 64


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

    Captures ``id(value)`` for every input, plus a sample content fingerprint
    for the types the fingerprint registry recognizes (pandas / numpy /
    mappings / sequences / other sized containers). Types with no fingerprint
    get identity-only tracking — reassignment is still detected, in-place
    mutation isn't.

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
        snapshots.append(
            InputSnapshot(
                var_name=var_name,
                identity=id(value),
                content_hash=_content_fingerprint(value),
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

    Compares the value's current content fingerprint against the one captured
    at snapshot time. A value with no fingerprint at snapshot time (unknown
    type) can't be checked and returns ``None``.

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
    if snapshot.content_hash is None:
        return None
    if _content_fingerprint(value) == snapshot.content_hash:
        return None
    return (
        f"'{snapshot.var_name}' was mutated in place (no reassignment)",
        "If a downstream cell needs the original, copy it before mutating "
        "(e.g. x = x.copy()) or reassign instead of mutating in place.",
    )


# ---------------------------------------------------------------------------
# Content fingerprint registry
#
# A value's fingerprint is a cheap, sampled digest taken before a cell runs and
# recompared after. The registry mirrors ``serializer._ARROW_TYPE_RULES``: an
# ordered list of (matches, fingerprint) pairs, first match wins. Adding a
# library is one entry. Fingerprints MUST stay cheap (sampled, not full-value)
# and MUST NOT raise — they run on every input of every cell execution, before
# the user's code; returning ``None`` means "couldn't fingerprint, skip".
# ---------------------------------------------------------------------------


class _FingerprintRule(NamedTuple):
    """One entry in the content-fingerprint registry.

    ``matches`` is a cheap, side-effect-free predicate; ``fingerprint`` returns
    a sampled hex digest or ``None`` when the value can't be hashed. NamedTuple
    (not ``@dataclass``) mirrors the serializer's registry tuples.
    """

    matches: collections.abc.Callable[[Any], bool]
    fingerprint: collections.abc.Callable[[Any], str | None]


def _content_fingerprint(value: Any) -> str | None:
    """Return a sampled content digest for *value*, or ``None`` if unknown.

    Walks :data:`_FINGERPRINT_RULES` in order; the first rule whose ``matches``
    accepts the value produces the digest. Unknown types yield ``None``
    (identity-only tracking).
    """
    for rule in _FINGERPRINT_RULES:
        if rule.matches(value):
            return rule.fingerprint(value)
    return None


def _is_pandas(value: Any) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False
    return isinstance(value, (pd.DataFrame, pd.Series))


def _hash_pandas_sample(value: Any) -> str | None:
    """Digest a DataFrame/Series from its shape, dtypes, and head/tail rows."""
    h = hashlib.sha256()
    h.update(str(value.shape).encode())
    try:
        h.update(str(value.dtypes.to_dict()).encode())
    except AttributeError:
        # Series have a single .dtype, not .dtypes.to_dict().
        h.update(str(value.dtype).encode())

    try:
        h.update(value.head(_MAX_SAMPLE).to_json().encode())
        if len(value) > _MAX_SAMPLE:
            h.update(value.tail(_MAX_SAMPLE).to_json().encode())
    except (ValueError, OverflowError, TypeError):
        # to_json() chokes on some object-dtype payloads; degrade gracefully.
        return None
    return h.hexdigest()


def _is_numpy(value: Any) -> bool:
    try:
        import numpy as np
    except ImportError:
        return False
    return isinstance(value, np.ndarray)


def _hash_ndarray_sample(value: Any) -> str | None:
    """Digest an ndarray from its shape, dtype, and a head/tail element sample."""
    import numpy as np

    h = hashlib.sha256()
    h.update(str(value.shape).encode())
    h.update(str(value.dtype).encode())
    try:
        flat = np.ascontiguousarray(value).reshape(-1)
        if flat.size > 2 * _MAX_SAMPLE:
            flat = np.concatenate([flat[:_MAX_SAMPLE], flat[-_MAX_SAMPLE:]])
        h.update(flat.tobytes())
    except (ValueError, TypeError):
        # Object/structured dtypes that won't reduce to bytes — skip.
        return None
    return h.hexdigest()


def _is_mapping(value: Any) -> bool:
    return isinstance(value, collections.abc.Mapping)


def _hash_mapping_sample(value: Any) -> str | None:
    """Digest a mapping from its length and a sorted sample of key reprs.

    Keys are hashable (so repr is safe and cheap); values are not hashed — a
    same-key value edit is the subscript form the static analyzer already
    recaptures (``d[k] = v``). This catches add / remove / clear / pop / update.
    """
    h = hashlib.sha256()
    try:
        h.update(str(len(value)).encode())
        for key in sorted(value.keys(), key=repr)[: 2 * _MAX_SAMPLE]:
            h.update(repr(key)[:_MAX_REPR].encode())
    except (TypeError, ValueError):
        return None
    return h.hexdigest()


def _is_sequence(value: Any) -> bool:
    # Concrete mutable/indexable sequences only — str/bytes are immutable, and
    # abc.Sequence would wrongly include them.
    return isinstance(value, (list, tuple))


def _hash_sequence_sample(value: Any) -> str | None:
    """Digest a list/tuple from its length and the identities of a head/tail
    sample of elements.

    ``id()`` (not ``repr``) keeps this crash-proof for arbitrary elements and
    is stable within the snapshot→detect window (same process). It catches
    append / extend / insert / remove / pop / sort / reverse; an in-place edit
    of an element object isn't a mutation of the sequence itself.
    """
    h = hashlib.sha256()
    try:
        n = len(value)
        h.update(str(n).encode())
        sample = value if n <= 2 * _MAX_SAMPLE else (*value[:_MAX_SAMPLE], *value[-_MAX_SAMPLE:])
        for element in sample:
            h.update(str(id(element)).encode())
    except (TypeError, ValueError):
        return None
    return h.hexdigest()


def _is_sized(value: Any) -> bool:
    # Last-resort catch-all for other sized containers (set, frozenset, deque,
    # custom). Earlier rules claim pandas/numpy/dict/list first. str/bytes are
    # immutable, so excluded.
    return isinstance(value, collections.abc.Sized) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _hash_len_only(value: Any) -> str | None:
    """Length-only digest — catches add/remove on otherwise-opaque containers."""
    try:
        return hashlib.sha256(str(len(value)).encode()).hexdigest()
    except (TypeError, ValueError):
        return None


# Order: concrete library types first, then the sized catch-all. torch / jax /
# polars are intentionally deferred (see design-mutation-fingerprint-registry).
_FINGERPRINT_RULES: tuple[_FingerprintRule, ...] = (
    _FingerprintRule(_is_pandas, _hash_pandas_sample),
    _FingerprintRule(_is_numpy, _hash_ndarray_sample),
    _FingerprintRule(_is_mapping, _hash_mapping_sample),
    _FingerprintRule(_is_sequence, _hash_sequence_sample),
    _FingerprintRule(_is_sized, _hash_len_only),
)


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
