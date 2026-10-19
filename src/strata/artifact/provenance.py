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

    @classmethod
    def from_json(cls, json_str: str) -> TransformSpec:
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(
            executor=data["executor"],
            params=data["params"],
            inputs=data["inputs"],
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
