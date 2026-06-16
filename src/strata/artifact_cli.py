"""Implementation of the ``strata artifact`` CLI subcommands.

Direct-store maintenance and inspection — no server required. The data
model already answers "what artifacts exist", "where did this come from",
and "which snapshot trained this model"; these commands render it.

Commands:
    list     Artifacts in the store (id, version, state, rows, size, names)
    show     One artifact's metadata, names, and direct inputs
    lineage  Walk provenance upstream to tables/snapshots
    pull     Write an artifact's blob to a local file
    verify   Check every blob against its metadata (see #123)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from strata.artifact_store import ArtifactStore, ArtifactVersion


def _open_store(artifact_dir_arg: str | None) -> ArtifactStore | None:
    artifact_dir = (
        Path(artifact_dir_arg) if artifact_dir_arg else Path.home() / ".strata" / "artifacts"
    )
    if not artifact_dir.exists():
        print(f"artifact directory not found: {artifact_dir}", file=sys.stderr)
        return None
    return ArtifactStore(artifact_dir)


class AmbiguousRefError(ValueError):
    """A name/alias resolves in more than one tenant — ``--tenant`` is required."""


def _tenant_matches(stored: str | None, requested: str) -> bool:
    return (stored or "") == requested


def _single_hit(ref: str, hits: list[Any]) -> Any | None:
    """Collapse name/alias hits to the single match, or ``None`` if none.

    Raises ``AmbiguousRefError`` when the hits span more than one tenant — there
    is at most one pointer per (tenant, name), so >1 hit means >1 tenant.
    """
    if not hits:
        return None
    tenants = {h.tenant or "" for h in hits}
    if len(tenants) > 1:
        listed = ", ".join(sorted(t or "<tenantless>" for t in tenants))
        raise AmbiguousRefError(
            f"{ref!r} exists in multiple tenants ({listed}); disambiguate with --tenant"
        )
    return hits[0]


def _resolve_ref(
    store: ArtifactStore, ref: str, tenant: str | None = None
) -> ArtifactVersion | None:
    """Resolve a CLI artifact reference.

    Accepted forms, tried in order: ``<id>@v=<N>``, a name pointer, ``name@alias``,
    and a bare artifact id (latest version). Name/alias lookups span tenants (a
    store inspector must find names whatever tenant wrote them, legacy "_default"
    included); pass ``tenant`` to scope to one.

    Raises:
        AmbiguousRefError: a name/alias matches in more than one tenant and no
            ``tenant`` was given — so the CLI can't silently inspect/pull the
            wrong tenant's artifact.
    """
    # id@v=N is tenant-independent.
    if "@v=" in ref:
        artifact_id, _, version_str = ref.partition("@v=")
        try:
            return store.get_artifact(artifact_id, int(version_str))
        except ValueError:
            return None

    # Name pointer.
    name_hits = [
        p
        for p in store.list_all_names()
        if p.name == ref and (tenant is None or _tenant_matches(p.tenant, tenant))
    ]
    hit = _single_hit(ref, name_hits)
    if hit is not None:
        return store.get_artifact(hit.artifact_id, hit.version)

    # name@alias (registry pointer): taxi/tip-model@champion
    if "@" in ref:
        name_part, _, alias_part = ref.rpartition("@")
        alias_hits = [
            a
            for a in store.list_all_aliases()
            if a.name == name_part
            and a.alias == alias_part
            and (tenant is None or _tenant_matches(a.tenant, tenant))
        ]
        hit = _single_hit(ref, alias_hits)
        if hit is not None:
            return store.get_artifact(hit.artifact_id, hit.version)

    # Bare artifact id → latest version.
    return store.get_latest_version(ref)


def _resolve_for_cmd(store: ArtifactStore, args: argparse.Namespace) -> ArtifactVersion | None:
    """Resolve ``args.ref`` honoring an optional ``--tenant``; print a message and
    return ``None`` on not-found or cross-tenant ambiguity."""
    try:
        artifact = _resolve_ref(store, args.ref, tenant=getattr(args, "tenant", None))
    except AmbiguousRefError as e:
        print(str(e), file=sys.stderr)
        return None
    if artifact is None:
        print(f"artifact not found: {args.ref}", file=sys.stderr)
    return artifact


def _names_for(store: ArtifactStore, artifact_id: str, version: int) -> list[str]:
    return [
        n.name
        for n in store.list_all_names()
        if n.artifact_id == artifact_id and n.version == version
    ]


def _fmt_when(created_at: float | None) -> str:
    if not created_at:
        return "-"
    return datetime.fromtimestamp(created_at, tz=UTC).strftime("%Y-%m-%d %H:%M")


def _fmt_size(byte_size: int | None) -> str:
    if byte_size is None:
        return "-"
    size = float(byte_size)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def _artifact_payload(store: ArtifactStore, artifact: ArtifactVersion) -> dict:
    input_versions = json.loads(artifact.input_versions) if artifact.input_versions else {}
    aliases = [
        f"{a.name}@{a.alias}"
        for a in store.list_all_aliases()
        if a.artifact_id == artifact.id and a.version == artifact.version
    ]
    return {
        "artifact_id": artifact.id,
        "version": artifact.version,
        "tenant": artifact.tenant or None,
        "principal": artifact.principal,
        "state": artifact.state,
        "row_count": artifact.row_count,
        "byte_size": artifact.byte_size,
        "created_at": artifact.created_at,
        "names": _names_for(store, artifact.id, artifact.version),
        "aliases": aliases,
        "tags": store.get_tags(artifact.id, artifact.version),
        "transform": json.loads(artifact.transform_spec) if artifact.transform_spec else None,
        "inputs": input_versions,
    }


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    store = _open_store(args.artifact_dir)
    if store is None:
        return 2

    artifacts = store.list_artifacts(limit=args.limit, state=args.state)
    if args.format == "json":
        print(
            json.dumps(
                [_artifact_payload(store, a) for a in artifacts],
                indent=2,
            )
        )
        return 0

    if not artifacts:
        print("no artifacts")
        return 0

    print(f"{'ID':<38} {'VER':>3} {'STATE':<10} {'ROWS':>10} {'SIZE':>8}  {'CREATED':<16} NAMES")
    for a in artifacts:
        names = ", ".join(_names_for(store, a.id, a.version))
        rows = f"{a.row_count:,}" if a.row_count is not None else "-"
        print(
            f"{a.id:<38} {a.version:>3} {a.state:<10} {rows:>10} "
            f"{_fmt_size(a.byte_size):>8}  {_fmt_when(a.created_at):<16} {names}"
        )
    return 0


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


def cmd_show(args: argparse.Namespace) -> int:
    store = _open_store(args.artifact_dir)
    if store is None:
        return 2
    artifact = _resolve_for_cmd(store, args)
    if artifact is None:
        return 1

    payload = _artifact_payload(store, artifact)
    if args.format == "json":
        print(json.dumps(payload, indent=2))
        return 0

    print(f"artifact:  {artifact.id}@v={artifact.version}")
    print(f"tenant:    {artifact.tenant or '-'}")
    print(f"state:     {artifact.state}")
    print(f"rows:      {artifact.row_count if artifact.row_count is not None else '-'}")
    print(f"size:      {_fmt_size(artifact.byte_size)}")
    print(f"created:   {_fmt_when(artifact.created_at)}")
    if payload["names"]:
        print(f"names:     {', '.join(payload['names'])}")
    if payload["aliases"]:
        print(f"aliases:   {', '.join(payload['aliases'])}")
    if payload["tags"]:
        print("tags:      " + ", ".join(f"{k}={v}" for k, v in payload["tags"].items()))
    transform = payload["transform"]
    if transform:
        print(f"transform: {transform.get('executor', '?')}")
    if payload["inputs"]:
        print("inputs:")
        for uri, version in payload["inputs"].items():
            print(f"  {uri}  ->  {version}")
    return 0


# ---------------------------------------------------------------------------
# lineage
# ---------------------------------------------------------------------------


def _walk_lineage(
    store: ArtifactStore,
    artifact: ArtifactVersion,
    *,
    max_depth: int,
    _depth: int = 0,
    _seen: set[tuple[str, int]] | None = None,
) -> dict:
    """Recursively resolve upstream provenance into a nested node dict."""
    seen = _seen if _seen is not None else set()
    seen.add((artifact.id, artifact.version))

    transform = json.loads(artifact.transform_spec) if artifact.transform_spec else {}
    node: dict = {
        "artifact_id": artifact.id,
        "version": artifact.version,
        "executor": transform.get("executor"),
        "names": _names_for(store, artifact.id, artifact.version),
        "inputs": [],
    }

    input_versions = json.loads(artifact.input_versions) if artifact.input_versions else {}
    for uri, version in input_versions.items():
        if uri.startswith("strata://artifact/") and _depth < max_depth:
            ref = uri[len("strata://artifact/") :]
            artifact_id, _, version_str = ref.partition("@v=")
            try:
                upstream = store.get_artifact(artifact_id, int(version_str))
            except ValueError:
                upstream = None
            if upstream is not None and (upstream.id, upstream.version) not in seen:
                node["inputs"].append(
                    _walk_lineage(
                        store, upstream, max_depth=max_depth, _depth=_depth + 1, _seen=seen
                    )
                )
                continue
        # Leaf: a table (version = snapshot id) or an unresolvable input
        node["inputs"].append({"uri": uri, "version": version})
    return node


def _render_lineage(node: dict, prefix: str = "", child_indent: str = "") -> None:
    """Render the lineage tree with box-drawing connectors."""
    if "artifact_id" in node:
        line = f"{node['artifact_id']}@v={node['version']}"
        if node.get("executor"):
            line += f"  [{node['executor']}]"
        if node.get("names"):
            line += f"  ({', '.join(node['names'])})"
    else:
        line = f"table {node['uri']}  @ snapshot {node['version']}"
    print(prefix + line)

    children = node.get("inputs", [])
    for i, child in enumerate(children):
        last = i == len(children) - 1
        _render_lineage(
            child,
            prefix=child_indent + ("└─ " if last else "├─ "),
            child_indent=child_indent + ("   " if last else "│  "),
        )


def cmd_lineage(args: argparse.Namespace) -> int:
    store = _open_store(args.artifact_dir)
    if store is None:
        return 2
    artifact = _resolve_for_cmd(store, args)
    if artifact is None:
        return 1

    tree = _walk_lineage(store, artifact, max_depth=args.max_depth)
    if args.format == "json":
        print(json.dumps(tree, indent=2))
    else:
        _render_lineage(tree)
    return 0


# ---------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------


def cmd_pull(args: argparse.Namespace) -> int:
    store = _open_store(args.artifact_dir)
    if store is None:
        return 2
    artifact = _resolve_for_cmd(store, args)
    if artifact is None:
        return 1
    if artifact.state not in ("ready", "superseded"):
        print(f"artifact is not readable (state={artifact.state})", file=sys.stderr)
        return 1

    blob = store.blob_store.read_blob(artifact.id, artifact.version)
    if blob is None:
        print(f"blob missing for {artifact.id}@v={artifact.version}", file=sys.stderr)
        return 1

    if args.to:
        out_path = Path(args.to)
    else:
        safe_ref = args.ref.replace("/", "_").replace("@", "_")
        out_path = Path(f"{safe_ref}.arrow")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob)
    print(f"wrote {len(blob):,} bytes to {out_path}  ({artifact.id}@v={artifact.version})")
    return 0


# ---------------------------------------------------------------------------
# verify (moved from cli.py for cohesion)
# ---------------------------------------------------------------------------


def cmd_verify(args: argparse.Namespace) -> int:
    store = _open_store(args.artifact_dir)
    if store is None:
        return 2

    findings = store.verify_artifacts()

    if args.format == "json":
        artifact_dir = (
            args.artifact_dir if args.artifact_dir else str(Path.home() / ".strata" / "artifacts")
        )
        print(json.dumps({"artifact_dir": artifact_dir, "findings": findings}, indent=2))
    else:
        print(f"verifying: {store.artifact_dir}")
        if not findings:
            print("\n✓ store is consistent")
        else:
            for f in findings:
                print(f"  ✗ {f['artifact_id']}@v={f['version']} [{f['problem']}] {f['detail']}")
            print(f"\n{len(findings)} problem(s) found")

    return 1 if findings else 0


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def cmd_audit(args: argparse.Namespace) -> int:
    """Render the append-only registry audit, newest first."""
    store = _open_store(args.artifact_dir)
    if store is None:
        return 2

    entries = store.read_audit(name=args.name, limit=args.limit)
    if args.format == "json":
        print(json.dumps(entries, indent=2))
        return 0

    if not entries:
        print("no audit entries")
        return 0

    for e in entries:
        when = _fmt_when(e["at"])
        actor = e["actor"] or "-"
        target = e["name"] or e["artifact_id"] or "?"
        if e["alias"]:
            target += f"@{e['alias']}"
        detail = ""
        if e["action"] in ("name_set", "alias_set"):
            to_ref = f"{(e['artifact_id'] or '?')[:8]}@v{e['to_version']}"
            if e.get("from_artifact_id"):
                from_ref = f"{e['from_artifact_id'][:8]}@v{e['from_version']}"
                detail = f"{from_ref} -> {to_ref}"
            elif e["from_version"] is not None:
                detail = f"v{e['from_version']} -> {to_ref}"
            else:
                detail = f"-> {to_ref}"
        elif e["action"] in ("name_delete", "alias_delete"):
            detail = f"was v{e['from_version']}" if e["from_version"] is not None else ""
        elif e["action"].startswith("tag_"):
            detail = f"{e['key']}={e['value']}" if e["value"] is not None else e["key"] or ""
        print(f"{when}  {e['action']:<12} {target:<40} {detail}  [{actor}]")
    return 0


def cmd_pending(args: argparse.Namespace) -> int:
    """List protected-alias changes awaiting approval."""
    store = _open_store(args.artifact_dir)
    if store is None:
        return 2

    entries = store.list_pending_changes()
    if args.format == "json":
        print(json.dumps(entries, indent=2))
        return 0
    if not entries:
        print("no pending changes")
        return 0
    for e in entries:
        target = f"{e['name']}@{e['alias']}"
        change = f"set -> {e['artifact_id']}@v={e['version']}" if e["action"] == "set" else "delete"
        requested_by = e["requested_by"] or "-"
        print(f"{_fmt_when(e['requested_at'])}  {target:<40} {change}  [{requested_by}]")
    return 0
