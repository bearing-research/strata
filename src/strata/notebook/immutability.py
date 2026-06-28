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
import sys
import types
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
    namespace: dict[str, Any],
    snapshots: list[InputSnapshot],
    exported_names: set[str] | None = None,
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

        # If the cell also exported this variable, downstream cells receive the
        # mutated value — it's a *published* mutation, not the dangerous silent
        # one. Only warn when a mutated input was NOT exported (→ downstream gets
        # the pre-mutation value).
        if exported_names is not None and snapshot.var_name in exported_names:
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
        f"'{snapshot.var_name}' was mutated in place (no reassignment); a "
        "downstream cell will see the pre-mutation value unless this cell "
        "exports it",
        "Reassign it (x = …), copy before mutating (x = x.copy()), or keep the "
        "producer and the mutation in one cell.",
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


# Types that can't be mutated in place — identity-only is correct, and
# fingerprinting them is wasted work (a str/int can't change under your feet).
_IMMUTABLE_SCALARS = (str, bytes, int, float, bool, complex, type(None))


def _general_fingerprint(value: Any) -> str | None:
    """Serializer-based fallback fingerprint for any picklable object.

    The :data:`_FINGERPRINT_RULES` above are *performance* optimizations for hot
    types (sampled digests). This is the *general* path that makes mutation
    detection cover arbitrary objects — ``torch.nn.Module``, sklearn estimators,
    custom classes — with **no per-type rule**, by hashing the same bytes Strata
    would store the value as. Immutable scalars are skipped; an unpicklable value
    falls back to identity-only (``None``).
    """
    if isinstance(value, _IMMUTABLE_SCALARS):
        return None
    try:
        import cloudpickle

        return hashlib.sha256(cloudpickle.dumps(value, protocol=5)).hexdigest()
    except Exception:
        # Unpicklable or non-deterministic to serialize → can't content-check.
        return None


def _content_fingerprint(value: Any) -> str | None:
    """Return a content digest for *value*, or ``None`` if it can't be checked.

    Walks :data:`_FINGERPRINT_RULES` (fast sampled digests for hot types) in
    order; the first match wins. Anything else falls back to a general
    serializer-based hash — so mutation detection isn't limited to a
    hand-written type registry.
    """
    for rule in _FINGERPRINT_RULES:
        if rule.matches(value):
            return rule.fingerprint(value)
    return _general_fingerprint(value)


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


def _is_torch(value: Any) -> bool:
    # A tensor implies torch is imported, so probe sys.modules rather than
    # importing torch (slow) to reject every value in torch-installed notebooks.
    # Mirrors serializer._matches_torch. (jax arrays are immutable by design —
    # no in-place mutation — so they need no rule.)
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(value, torch.Tensor)


def _hash_torch_sample(value: Any) -> str | None:
    """Digest a torch tensor from shape/dtype/device + a detached element sample.

    Slices the sample in torch *before* converting to numpy, so huge tensors
    aren't fully materialized. Exotic dtypes that ``.numpy()`` refuses (bf16,
    quantized) fall back to identity-only by returning ``None``. Catches the
    trailing-underscore in-place ops (``x.add_()``, ``x.zero_()``, …).
    """
    h = hashlib.sha256()
    h.update(str(tuple(value.shape)).encode())
    h.update(str(value.dtype).encode())
    h.update(str(value.device).encode())
    try:
        flat = value.detach().flatten()
        n = int(flat.shape[0])
        parts = [flat] if n <= 2 * _MAX_SAMPLE else [flat[:_MAX_SAMPLE], flat[-_MAX_SAMPLE:]]
        for part in parts:
            h.update(part.cpu().numpy().tobytes())
    except (TypeError, RuntimeError, ValueError):
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


# Order: concrete library types first, then the sized catch-all. polars is
# deferred (its API is mostly immutable); jax needs no rule (jax arrays are
# immutable). See design-mutation-fingerprint-registry.
_FINGERPRINT_RULES: tuple[_FingerprintRule, ...] = (
    _FingerprintRule(_is_pandas, _hash_pandas_sample),
    _FingerprintRule(_is_numpy, _hash_ndarray_sample),
    _FingerprintRule(_is_torch, _hash_torch_sample),
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


# ---------------------------------------------------------------------------
# Shared-mutable-object detection across a cell's outputs
#
# Strata stores each output variable as an independent artifact. If two outputs
# share a mutable object by identity — the classic case being an optimizer that
# holds a model's parameter tensors — storing them separately *decouples* them:
# downstream they're independent copies, so mutating one no longer affects the
# other. That silently breaks split model/optimizer training. Detection rides a
# bounded object-graph walk (no per-library rule); arrays/tensors are recorded
# as mutable leaves but not traversed into.
# ---------------------------------------------------------------------------


# Shared by nature (imports, defs) and never the "decoupling-relevant" state we
# care about — skip recording and traversing them, or two outputs that both
# reference numpy would look like they "share" the module.
_SHARED_BY_NATURE = (
    types.ModuleType,
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    type,
)


def _is_opaque_leaf(value: Any) -> bool:
    """A mutable object we record but don't traverse into (huge buffers)."""
    return _is_numpy(value) or _is_torch(value) or _is_pandas(value)


def _reachable_mutable_ids(
    root: Any, *, max_nodes: int = 20000, max_depth: int = 8
) -> dict[int, type]:
    """Map ``id -> type`` for mutable objects reachable from *root* (bounded).

    Immutable scalars are ignored; immutable containers (tuple/frozenset) are
    traversed but not recorded; arrays/tensors are recorded but not traversed
    into. Only ``__dict__`` is followed on custom objects — never ``__slots__``
    descriptors, which could trigger side effects.
    """
    found: dict[int, type] = {}
    visited: set[int] = set()
    stack: list[tuple[Any, int]] = [(root, 0)]
    while stack and len(visited) < max_nodes:
        obj, depth = stack.pop()
        oid = id(obj)
        if oid in visited or depth > max_depth:
            continue
        visited.add(oid)
        if isinstance(obj, (_IMMUTABLE_SCALARS, _SHARED_BY_NATURE)):
            continue
        if isinstance(obj, (tuple, frozenset)):
            for child in obj:
                stack.append((child, depth + 1))
            continue
        found[oid] = type(obj)
        if _is_opaque_leaf(obj):
            continue
        try:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    stack.append((key, depth + 1))
                    stack.append((value, depth + 1))
            elif isinstance(obj, (list, set)):
                for child in obj:
                    stack.append((child, depth + 1))
            else:
                obj_dict = getattr(obj, "__dict__", None)
                if isinstance(obj_dict, dict):
                    for value in obj_dict.values():
                        stack.append((value, depth + 1))
        except Exception:
            # Exotic container / proxy whose iteration raised — stop descending
            # this branch rather than abort the whole walk.
            continue
    return found


def detect_shared_mutable_outputs(outputs: dict[str, Any]) -> list[MutationWarning]:
    """Warn when two of a cell's outputs share a mutable object by identity.

    Such outputs decouple once stored as separate artifacts (see the module
    note above). General — no per-type rule — it walks each output's object
    graph and reports the first shared mutable object per output pair.
    """
    owners: dict[int, str] = {}
    reported: set[frozenset[str]] = set()
    warnings: list[MutationWarning] = []
    for var_name, value in outputs.items():
        for oid, typ in _reachable_mutable_ids(value).items():
            prev = owners.get(oid)
            if prev is None:
                owners[oid] = var_name
            elif prev != var_name:
                pair = frozenset((prev, var_name))
                if pair in reported:
                    continue
                reported.add(pair)
                warnings.append(
                    MutationWarning(
                        var_name=prev,
                        message=(
                            f"outputs '{prev}' and '{var_name}' share a mutable "
                            f"{typ.__name__} object; stored as separate artifacts "
                            "they become independent copies downstream"
                        ),
                        suggestion=(
                            "If they must stay linked (e.g. an optimizer over a "
                            f"model's parameters), keep '{prev}' and '{var_name}' "
                            "in the same cell."
                        ),
                    )
                )
    return warnings
