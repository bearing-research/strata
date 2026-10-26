"""Transform specs and the provenance hash that deduplicates artifacts by them.

Pure: no store or server access, so any layer can build or hash a spec.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class TransformSpec:
    """Transform specification: stored by the server, interpreted only by the executor.

    ``executor`` is a URI such as ``local://duckdb_sql@v1``; ``inputs`` are table
    or artifact URIs.
    """

    executor: str
    params: dict
    inputs: list[str]

    @cached_property
    def canonical_json(self) -> str:
        """The spec as stored and hashed; cached, so ``params`` must not change after use.

        These bytes feed every provenance hash, so their format must not change.
        Inputs keep their caller order: positional executors bind them as ``input0,
        input1, ...``, so ``f(a, b)`` must not dedup against ``f(b, a)``.
        """
        return json.dumps(
            {
                "executor": self.executor,
                "params": self.params,
                "inputs": list(self.inputs),
            },
            sort_keys=True,
        )


def parse_transform_spec(text: str | None) -> TransformSpec | None:
    """A stored spec read back; ``None`` when there is none or it is not a JSON object.

    Unreadable reads as unrecorded, not an error: rows also arrive from other servers
    (team store pulls, snapshot imports, datasets), and every reader treats a spec it
    cannot read as one that names nothing. A missing or mistyped field reads as empty.
    """
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    params = data.get("params")
    inputs = data.get("inputs")
    return TransformSpec(
        executor=data.get("executor") or "",
        params=params if isinstance(params, dict) else {},
        inputs=inputs if isinstance(inputs, list) else [],
    )


def compute_provenance_hash(input_hashes: list[str], transform_spec: TransformSpec) -> str:
    """SHA-256 of the sorted input hashes plus the transform spec JSON, for dedup.

    Input hashes are sorted, but the spec's own ``inputs`` keep caller order.
    """
    sorted_inputs = sorted(input_hashes)

    hasher = hashlib.sha256()
    for h in sorted_inputs:
        hasher.update(h.encode("utf-8"))
        hasher.update(b"\x00")  # Separator
    hasher.update(transform_spec.canonical_json.encode("utf-8"))

    return hasher.hexdigest()


def derive_subkey(parent_hash: str, *labels: str) -> str:
    """Derive ``sha256("parent_hash:label1:label2:...")`` from a provenance hash.

    Namespaces one computation's outputs: per variable, per display, per iteration
    and, under ADR 002 §7.1, per build and release. The byte format is wire-stable:
    changing it invalidates every cached artifact keyed off a derived hash.
    """
    pieces = [parent_hash, *labels]
    return hashlib.sha256(":".join(pieces).encode()).hexdigest()


def checkpoint_output_key(parent_hash: str, build_id: str, sequence: int, output: str) -> str:
    """The key of one output of a ``checkpoint`` release (ADR 002 §7.1).

    Labelled by build and release so every checkpoint of a bound computation keeps
    its own identity, and none is ever the cache hit for the computation itself.
    """
    return derive_subkey(parent_hash, f"build={build_id}", f"release={sequence}", output)


def run_output_key(parent_hash: str, build_id: str, output: str) -> str:
    """The key of a final output persisted per run (a stochastic computation)."""
    return derive_subkey(parent_hash, f"build={build_id}", output)
