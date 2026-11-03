"""Module export: defs and classes a downstream cell reads, written as synthetic artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vestigo.provenance import derive_subkey

from strata.notebook.harness_user import read_run_file, write_run_file
from strata.notebook.module_export import build_module_export_plan, runtime_binding_names
from strata.notebook.provenance import compute_source_hash, safe_filename_stem

if TYPE_CHECKING:
    from strata.notebook.executor_host import ExecutorHost


def write_module_export_outputs(
    host: ExecutorHost,
    cell_id: str,
    source: str,
    output_dir: Path,
    provenance_hash: str,
    outputs: dict[str, Any],
) -> str | None:
    """Write synthetic module artifacts for cross-cell defs/classes.

    Returns an error string when a downstream-consumed definition cannot be
    exported safely.
    """
    if host.dag is None:
        return None

    consumed_vars = host.dag.consumed_variables.get(cell_id, set())
    if not consumed_vars:
        return None

    # Upstream-produced names can be hydrated into the synthetic module at load time
    # instead of blocking a def/class that closes over them.
    producer = host.dag.variable_producer
    cells_by_id = {c.id: c for c in host.notebook_state.cells}
    this_cell = cells_by_id.get(cell_id)
    cross_cell = frozenset(
        v
        for v in ([*this_cell.references, *this_cell.builtin_references] if this_cell else [])
        if isinstance(producer.get(v), str) and producer[v] != cell_id
    )
    # Same-cell runtime values (a loaded model, a cwd-derived path) can be hydrated too.
    same_cell_runtime = runtime_binding_names(source)
    injectable = cross_cell | same_cell_runtime

    export_plan = build_module_export_plan(source, injectable=injectable)
    exportable_vars = sorted(set(export_plan.exported_symbols) & set(consumed_vars))
    blocked_vars = sorted(export_plan.blocking_symbols & set(consumed_vars))
    if not exportable_vars and not blocked_vars:
        return None

    if not export_plan.is_exportable:
        joined_vars = ", ".join(sorted(set(exportable_vars) | set(blocked_vars)))
        return (
            "This cell defines reusable code used downstream "
            f"({joined_vars}), but it cannot be shared across cells yet: "
            f"{export_plan.format_error()}"
        )

    # Constants alone don't trigger module-export (``x = 1`` stays a plain int). They
    # ride the module only when the cell also exports a def/class.
    code_exports = [
        name
        for name in exportable_vars
        if export_plan.exported_symbols[name].kind in ("function", "async function", "class")
    ]
    if not code_exports:
        return None

    # Pin each injected name to the upstream variable's versioned artifact URI. If one
    # can't be pinned, block rather than ship a module that NameErrors at call time.
    injected_refs: dict[str, str] = {}
    for name in sorted(export_plan.injected_inputs):
        prod_id = producer.get(name)
        if isinstance(prod_id, str) and prod_id != cell_id:
            prod_cell = cells_by_id.get(prod_id)
            uri = prod_cell.artifact_uris.get(name) if prod_cell is not None else None
        else:
            # Same-cell: persist the runtime value now (store is idempotent by provenance, so
            # the later _store_outputs pass is a no-op) and pin its URI.
            uri = _store_same_cell_injected(
                host, cell_id, name, output_dir, outputs, provenance_hash, source
            )
        if uri is None:
            return (
                "This cell defines reusable code used downstream "
                f"({', '.join(sorted(set(exportable_vars) | set(blocked_vars)))}), "
                f"but a value it closes over (`{name}`) isn't available to "
                "share yet."
            )
        injected_refs[name] = uri

    source_hash = compute_source_hash(source)
    notebook_id = host.notebook_state.id
    # Fold the injected identity into the module name so sys.modules can't alias two
    # hydrations of the same slice.
    injected_tag = hashlib.sha256(
        "|".join(f"{k}={v}" for k, v in sorted(injected_refs.items())).encode()
    ).hexdigest()

    module_suffix = source_hash[:12]
    if injected_refs:
        module_suffix = f"{source_hash[:12]}_{injected_tag[:8]}"

    for var_name in exportable_vars:
        symbol = export_plan.exported_symbols[var_name]
        descriptor = {
            "module_name": (f"nb_{notebook_id}_{cell_id}_{var_name}_{module_suffix}"),
            "symbol_name": var_name,
            "kind": symbol.kind,
            "source": export_plan.module_source,
            "provenance_hash": derive_subkey(provenance_hash, var_name),
            "injected": injected_refs,
        }
        file_name = f"{safe_filename_stem(var_name)}.cell_module.json"
        data = json.dumps(descriptor).encode("utf-8")
        write_run_file(output_dir, file_name, data)

        outputs[var_name] = {
            "content_type": "module/cell",
            "file": file_name,
            "bytes": len(data),
            "type": symbol.kind,
            "preview": f"<{symbol.kind} {var_name}>",
        }

    return None


def _store_same_cell_injected(
    host: ExecutorHost,
    cell_id: str,
    var_name: str,
    output_dir: Path,
    outputs: dict[str, Any],
    provenance_hash: str,
    source: str,
) -> str | None:
    """Store a same-cell value an exported def closes over; return its pinned URI or None.

    The harness already wrote its blob to *output_dir*. ``store_cell_output``
    dedups by provenance, so a later ``_store_outputs`` of it is a no-op.
    """
    spec = outputs.get(var_name)
    if not isinstance(spec, dict) or "file" not in spec or "error" in spec:
        return None
    blob_path = output_dir / spec["file"]
    if not blob_path.exists():
        return None
    artifact_mgr = host.get_artifact_manager()
    artifact_version = artifact_mgr.store_cell_output(
        cell_id=cell_id,
        variable_name=var_name,
        blob_data=read_run_file(output_dir, str(spec["file"])),
        content_type=spec.get("content_type", "pickle/object"),
        provenance_hash=derive_subkey(provenance_hash, var_name),
        source_hash=compute_source_hash(source),
        source=source,
    )
    return f"strata://artifact/{artifact_version.id}@v={artifact_version.version}"
