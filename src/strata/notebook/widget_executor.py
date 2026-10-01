"""Executor for widget cells: one value artifact per control, no subprocess.

Each control's value comes from ``runtime.json`` (``CellRuntime.widget_values``)
or its declared default, and is stored as a ``json/object`` scalar under
``nb_{notebook}_cell_{cell}_var_{name}`` with a per-value provenance hash.
"""

from __future__ import annotations

import json
import time
from typing import Any

from strata.notebook.provenance import derive_subkey
from strata.notebook.widget_analyzer import analyze_widget_cell, descriptor_provenance


def _current_values(session: Any, cell_id: str) -> dict[str, Any]:
    """The cell's user-set control values from ``runtime.json``."""
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
                source=source,
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
