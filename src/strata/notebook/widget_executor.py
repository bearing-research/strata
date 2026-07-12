"""Executor for widget cells (P2).

A widget cell produces one value artifact per declared control — with **no
subprocess**. It is the simplest instance of the "produce an artifact without
running user code" pattern that prompt cells established:

- The current value of each control comes from ``runtime.json``
  (``CellRuntime.widget_values``), falling back to the declared default.
- Each value is stored as a ``json/object`` scalar under the canonical id
  ``nb_{notebook}_cell_{cell}_var_{name}``, keyed by a per-value provenance
  hash (declaration + value). Downstream cells resolve it like any upstream
  output; returning a control to a prior value reproduces the hash, so
  downstream re-computation is a cache hit.

There is no LLM call, no network, no harness — this is pure store I/O.
"""

from __future__ import annotations

import json
import time
from typing import Any

from strata.notebook.provenance import derive_subkey
from strata.notebook.widget_analyzer import analyze_widget_cell, descriptor_provenance


def _current_values(session: Any, cell_id: str) -> dict[str, Any]:
    """The cell's user-set control values from ``runtime.json`` (P3 writes them)."""
    from strata.notebook.runtime_state import load_runtime_state

    runtime = load_runtime_state(session.path)
    entry = runtime.cells.get(cell_id)
    return dict(entry.widget_values) if entry and entry.widget_values else {}


def execute_widget_cell(
    session: Any,
    cell_id: str,
    source: str,
    *,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Materialize each control's current value as a cached artifact."""
    start_time = time.time()
    analysis = analyze_widget_cell(source)
    if analysis.errors:
        return {
            "success": False,
            "error": "; ".join(analysis.errors),
            "outputs": {},
            "display_outputs": [],
            "cache_hit": False,
            "duration_ms": int((time.time() - start_time) * 1000),
            "execution_method": "widget",
        }

    artifact_mgr = session.get_artifact_manager()
    notebook_id = session.notebook_state.id
    values = _current_values(session, cell_id)

    outputs: dict[str, Any] = {}
    resolved: dict[str, Any] = {}
    all_cache_hits = True

    for descriptor in analysis.descriptors:
        name = descriptor.name
        value = values.get(name, descriptor.default)
        resolved[name] = value

        var_provenance = derive_subkey(descriptor_provenance(descriptor, value), name)
        canonical_id = f"nb_{notebook_id}_cell_{cell_id}_var_{name}"

        canonical = artifact_mgr.artifact_store.get_latest_version(canonical_id)
        if use_cache and canonical is not None and canonical.provenance_hash == var_provenance:
            version = canonical.version
        else:
            all_cache_hits = False
            blob = json.dumps(value, default=str).encode()
            stored = artifact_mgr.store_cell_output(
                cell_id=cell_id,
                variable_name=name,
                blob_data=blob,
                content_type="json/object",
                row_count=1,
                provenance_hash=var_provenance,
            )
            version = stored.version

        outputs[name] = {
            "content_type": "json/object",
            "artifact_uri": f"strata://artifact/{canonical_id}@v={version}",
            "preview": value,
        }

    return {
        "success": True,
        "outputs": outputs,
        "display_outputs": [],
        "artifact_uri": next(iter(outputs.values()), {}).get("artifact_uri") if outputs else None,
        "cache_hit": all_cache_hits and bool(outputs),
        "duration_ms": int((time.time() - start_time) * 1000),
        "execution_method": "widget",
        "values": resolved,
    }
