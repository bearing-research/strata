"""Runtime mutation detection for notebook cell inputs.

Best-effort and warn-only: catches in-place mutation the static analyzer cannot
see (aliases, helper functions, bare method mutators). Each input gets an
identity check plus a sampled content fingerprint from an extensible registry;
see ``docs/internal/design-mutation-fingerprint-registry.md``.
"""

from __future__ import annotations

import collections.abc
import copy
import hashlib
import sys
import types
from dataclasses import dataclass
from typing import Any, NamedTuple, TypedDict

# Fingerprints sample head/tail only: this runs on every input of every cell execution.
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

    A TypedDict because its only consumers are JSON writers (manifest.json, WS
    ``cell_output`` payloads).
    """

    var_name: str
    message: str
    suggestion: str | None


def snapshot_inputs(namespace: dict[str, Any], input_names: list[str]) -> list[InputSnapshot]:
    """Snapshot identity and content fingerprint of each input present in *namespace*.

    A value with no fingerprint gets identity-only tracking: reassignment is still
    detected, in-place mutation is not.
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
    """Compare *namespace* after execution against *snapshots* and report mutations.

    A deleted input is reported; a reassigned one is not, since the input object was
    untouched. A same-identity value is reported when its fingerprint changed.
    Inputs in *exported_names* are skipped: downstream receives them as published.
    """
    warnings = []

    for snapshot in snapshots:
        if snapshot.var_name not in namespace:
            # Deleted counts as a mutation.
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

        # Reassigned, not mutated.
        if current_id != snapshot.identity:
            continue

        # An exported mutated input reaches downstream as published. Warn only when it was not
        # exported, since downstream then gets the pre-mutation value.
        if exported_names is not None and snapshot.var_name in exported_names:
            continue

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
    """Return ``(message, suggestion)`` if a same-identity *value* changed, else None.

    A value that had no fingerprint at snapshot time cannot be checked.
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
# A cheap sampled digest taken before a cell runs and recompared after. Ordered
# (matches, fingerprint) pairs, first match wins (mirrors
# ``serializer._ARROW_TYPE_RULES``). Fingerprints MUST stay cheap and MUST NOT raise:
# they run on every input before the user's code. ``None`` means "skip".
# ---------------------------------------------------------------------------


class _FingerprintRule(NamedTuple):
    """One entry in the content-fingerprint registry.

    ``matches`` is a cheap, side-effect-free predicate; ``fingerprint`` returns a
    sampled hex digest or ``None`` when the value cannot be hashed.
    """

    matches: collections.abc.Callable[[Any], bool]
    fingerprint: collections.abc.Callable[[Any], str | None]


# Can't be mutated in place, so identity-only is correct and fingerprinting is waste.
_IMMUTABLE_SCALARS = (str, bytes, int, float, bool, complex, type(None))


def _general_fingerprint(value: Any) -> str | None:
    """Fallback fingerprint: a hash of the bytes Strata would store *value* as.

    Covers arbitrary objects (modules, estimators, custom classes) with no per-type
    rule. Immutable scalars are skipped; an unpicklable value returns None
    (identity-only).
    """
    if isinstance(value, _IMMUTABLE_SCALARS):
        return None
    try:
        import cloudpickle

        return hashlib.sha256(cloudpickle.dumps(value, protocol=5)).hexdigest()
    except Exception:
        # Unpicklable or non-deterministic to serialize: can't content-check.
        return None


def _content_fingerprint(value: Any) -> str | None:
    """Return a content digest for *value*, or ``None`` if it cannot be checked.

    The first matching :data:`_FINGERPRINT_RULES` entry wins; anything else falls
    back to :func:`_general_fingerprint`.
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
        # to_json() chokes on some object-dtype payloads.
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
        # Object/structured dtypes that won't reduce to bytes.
        return None
    return h.hexdigest()


def _is_torch(value: Any) -> bool:
    # Probe sys.modules (a tensor implies torch is imported) instead of importing torch,
    # which is slow. Mirrors serializer._matches_torch.
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(value, torch.Tensor)


def _hash_torch_sample(value: Any) -> str | None:
    """Digest a torch tensor from shape/dtype/device and a detached element sample.

    Slices in torch before converting to numpy so huge tensors are not materialized.
    Dtypes ``.numpy()`` refuses (bf16, quantized) return None (identity-only).
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

    Values are not hashed: a same-key edit (``d[k] = v``) is already recaptured by
    the static analyzer.
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
    # Concrete sequences only: abc.Sequence would include immutable str/bytes.
    return isinstance(value, (list, tuple))


def _hash_sequence_sample(value: Any) -> str | None:
    """Digest a list/tuple from its length and the ids of a head/tail element sample.

    ``id()`` rather than ``repr`` cannot raise and is stable within one process. An
    in-place edit of an element is not a mutation of the sequence.
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
    # Catch-all for other sized containers (set, deque, custom); earlier rules claim
    # pandas/numpy/dict/list first. str/bytes are immutable, so excluded.
    return isinstance(value, collections.abc.Sized) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _hash_len_only(value: Any) -> str | None:
    """Length-only digest: catches add/remove on otherwise-opaque containers."""
    try:
        return hashlib.sha256(str(len(value)).encode()).hexdigest()
    except (TypeError, ValueError):
        return None


# Concrete library types first, then the sized catch-all. polars is deferred (mostly
# immutable API); jax arrays are immutable and need no rule.
_FINGERPRINT_RULES: tuple[_FingerprintRule, ...] = (
    _FingerprintRule(_is_pandas, _hash_pandas_sample),
    _FingerprintRule(_is_numpy, _hash_ndarray_sample),
    _FingerprintRule(_is_torch, _hash_torch_sample),
    _FingerprintRule(_is_mapping, _hash_mapping_sample),
    _FingerprintRule(_is_sequence, _hash_sequence_sample),
    _FingerprintRule(_is_sized, _hash_len_only),
)


def apply_defensive_copy(value: Any, content_type: str) -> Any:
    """Return a defensive copy of *value*, chosen by content type.

    Not wired into execution (inputs are re-read from the store each run); kept for
    opt-in input isolation. Content types are literal strings because this module
    runs in the notebook venv and cannot import ``ContentType``. ``arrow/ipc`` is not
    copied, ``json/object`` is shallow-copied, ``pickle/object`` is deep-copied.
    """
    if content_type == "json/object":
        return copy.copy(value)
    if content_type == "pickle/object":
        return copy.deepcopy(value)
    # arrow/ipc (fresh on deserialize) or unknown.
    return value


# ---------------------------------------------------------------------------
# Shared-mutable-object detection across a cell's outputs
#
# Each output is stored as an independent artifact, so two outputs sharing a mutable
# object (an optimizer holding a model's parameter tensors) come back decoupled
# downstream, silently breaking split model/optimizer training. Detection is a bounded
# object-graph walk; arrays/tensors are recorded as mutable leaves, not traversed.
# ---------------------------------------------------------------------------


# Shared by nature (imports, defs): skip them, or two outputs that both reference
# numpy would look like they share the module.
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

    Immutable containers are traversed but not recorded; arrays/tensors are recorded
    but not traversed. Only ``__dict__`` is followed on custom objects, never
    ``__slots__`` descriptors, which could have side effects.
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
            # Exotic container whose iteration raised: stop descending this branch only.
            continue
    return found


def detect_shared_mutable_outputs(outputs: dict[str, Any]) -> list[MutationWarning]:
    """Warn when two of a cell's outputs share a mutable object by identity.

    Such outputs become independent copies once stored as separate artifacts.
    Reports the first shared object per output pair.
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
                            "model's parameters), output only one and derive the "
                            "other from it in the cell that uses both."
                        ),
                    )
                )
    return warnings
