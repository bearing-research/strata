"""Implementation of the ``strata artifact`` CLI subcommands.

These work directly on the artifact store; no server is required.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strata.artifact_store import ArtifactStore, ArtifactVersion
from strata.artifact_transfer import (
    PublicationTarget,
    RemoteStore,
    copy_chain,
    promote_artifact,
)

if TYPE_CHECKING:
    from strata.config import StrataConfig


def _open_store(artifact_dir_arg: str | None) -> ArtifactStore | None:
    """Open the store a command reads.

    ``--artifact-dir`` names one local store (SQLite file plus blobs) and nothing else is
    consulted: a DSN or blob backend from the environment would pair this store's rows with
    another store's bytes. Without it, open the store the server is configured with
    (``[tool.strata]``, then ``STRATA_*``): its ``artifact_dir``, metadata DSN and blob backend.
    """
    if artifact_dir_arg:
        artifact_dir = Path(artifact_dir_arg)
        if not (artifact_dir / "artifacts.sqlite").exists():
            print(f"no artifact store in {artifact_dir}", file=sys.stderr)
            return None
        return ArtifactStore(artifact_dir)

    from strata.config import StrataConfig

    try:
        config = StrataConfig.load()
    except ValueError as exc:
        print(f"invalid configuration: {exc}", file=sys.stderr)
        return None
    artifact_dir = config.artifact_dir or Path.home() / ".strata" / "artifacts"
    # Loading the config creates a personal-mode artifact_dir, so an empty
    # directory proves nothing: a store on SQLite is its database file.
    if not config.artifact_metadata_dsn and not (artifact_dir / "artifacts.sqlite").exists():
        print(f"no artifact store in {artifact_dir}", file=sys.stderr)
        return None
    return _configured_store(config, artifact_dir)


def _configured_store(config: StrataConfig, artifact_dir: Path) -> ArtifactStore:
    """The store *config* describes under *artifact_dir*: its metadata DSN and blob backend."""
    blob_store = config.create_blob_store() if config.artifact_blob_backend != "local" else None
    return ArtifactStore(
        artifact_dir, blob_store=blob_store, dialect=config.create_metadata_dialect()
    )


class AmbiguousRefError(ValueError):
    """A name/alias resolves in more than one tenant; ``--tenant`` is required."""


def _tenant_matches(stored: str | None, requested: str) -> bool:
    return (stored or "") == requested


def _single_hit(ref: str, hits: list[Any]) -> Any | None:
    """Collapse name/alias hits to the single match, or ``None`` if none.

    Raises ``AmbiguousRefError`` when hits span tenants (at most one pointer per tenant and
    name, so more than one hit means more than one tenant).
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

    Forms, tried in order: ``<id>@v=<N>``, a name pointer, ``name@alias``, and a bare artifact
    id (latest version). Name/alias lookups span all tenants unless ``tenant`` is given.

    Raises:
        AmbiguousRefError: a name/alias matches in several tenants and no ``tenant`` was
            given, so the CLI never silently picks the wrong tenant's artifact.
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
    """Resolve ``args.ref`` honoring ``--tenant``.

    Prints a message and returns ``None`` on not-found or cross-tenant ambiguity.
    """
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


# --- list ---


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


# --- show ---


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


# --- lineage ---


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
        # A name input (a notebook's ``@dataset``) records the version it
        # resolved to, which is what the walk follows.
        if uri.startswith("strata://artifact/"):
            ref = uri[len("strata://artifact/") :]
        elif uri.startswith("strata://name/"):
            ref = str(version)
        else:
            ref = ""
        if "@v=" in ref and _depth < max_depth:
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

    tree = _walk_lineage(store, artifact, max_depth=getattr(args, "max_depth", 10))
    if args.format == "json":
        print(json.dumps(tree, indent=2))
    else:
        _render_lineage(tree)
    return 0


# --- publish / promote / export / unpublish ---


def _server_store() -> ArtifactStore | None:
    """The store the running server serves from, or ``None`` if unresolvable.

    Built as the server builds it, so a link minted here resolves there.
    """
    from strata.config import StrataConfig

    config = StrataConfig.load()
    return _configured_store(config, config.artifact_dir) if config.artifact_dir else None


def _publication_target(
    args: argparse.Namespace, source: ArtifactStore
) -> tuple[PublicationTarget, str]:
    """Where the grant is minted, and how to describe that to the caller.

    ``--artifact-dir`` says where to read from; a publication is written to the store the
    server serves, since only there does the link resolve. The destination is always named
    to the caller, whether or not it differs from the source.
    """
    to_url = getattr(args, "to_url", None)
    if to_url:
        # A store on another machine: the chain travels over HTTP and the grant is minted there,
        # since a link only resolves from the store that serves it. Remap, so a chain another
        # tenant already copied there lands as this caller's own copy instead of a 409.
        return RemoteStore(str(to_url), _remote_headers(args), remap=True), str(to_url)

    into = getattr(args, "into", None)
    if into:
        return ArtifactStore(Path(into)), str(into)

    if getattr(args, "here", False):
        return source, str(source.artifact_dir)

    server_store = _server_store()
    if server_store is None:
        # Nothing configured to serve from: publish in place and say so rather than mint a link that
        # resolves nowhere.
        return source, f"{source.artifact_dir} (no server store is configured)"
    return server_store, f"{server_store.artifact_dir} (the store your server serves)"


def _remote_headers(args: argparse.Namespace) -> dict[str, str]:
    """Auth for the remote store, from ``--header`` or the environment.

    Env by default so a token stays out of shell history and process lists.
    """
    headers: dict[str, str] = {}
    token = os.environ.get("STRATA_STORE_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for raw in getattr(args, "header", None) or []:
        name, _, value = str(raw).partition(":")
        if not value.strip():
            raise ValueError(f"--header expects 'Name: value', got {raw!r}")
        headers[name.strip()] = value.strip()
    return headers


def _parse_tags(raw_tags: list[str] | None) -> dict[str, str]:
    """Parse ``key=value`` strings into a dict, printing a warning for each malformed one."""
    tags: dict[str, str] = {}
    for raw in raw_tags or []:
        key, _, value = str(raw).partition("=")
        if not value:
            print(f"Ignoring malformed tag {raw!r}; expected key=value")
            continue
        tags[key.strip()] = value.strip()
    return tags


def cmd_promote(args: argparse.Namespace) -> int:
    """``strata artifact promote``: share a result with the team, on purpose.

    The walk and registry writes live in :func:`promote_artifact`; this resolves the ref,
    renders the outcome and picks an exit code.
    """
    store = _open_store(args.artifact_dir)
    if store is None:
        return 2
    artifact = _resolve_for_cmd(store, args)
    if artifact is None:
        return 1

    target = RemoteStore(str(args.to_url), _remote_headers(args))
    try:
        promotion = promote_artifact(
            store,
            target,
            artifact,
            name=args.name,
            alias=getattr(args, "alias", None),
            tags=_parse_tags(getattr(args, "tag", None)),
            max_depth=getattr(args, "max_depth", 10),
            table=getattr(args, "table", None),
        )
    except ValueError as exc:
        print(f"Cannot promote: {exc}")
        return 1
    except RuntimeError as exc:
        # What copied is harmless and reusable: it is keyed by provenance, so it is a cache entry
        # whether or not it got a name.
        print(f"Promotion failed partway: {exc}")
        return 1

    if args.format == "json":
        print(
            json.dumps(
                {
                    "name": promotion.name,
                    "artifact_uri": f"strata://artifact/{promotion.ref}",
                    "copied": promotion.copied,
                    "alias_pending": promotion.alias_pending,
                    "table": promotion.table,
                    "table_snapshot": promotion.table_snapshot,
                },
                indent=2,
            )
        )
        return 0

    print(f"Promoted {artifact.id}@v={artifact.version} to {args.to_url}")
    if promotion.ref != f"{artifact.id}@v={artifact.version}":
        # It deduplicated onto a row the store already had: the same
        # computation, promoted by someone else or offered by the cache.
        print(f"  the store already held this computation as {promotion.ref}")
    print(f"  name:  {promotion.name} -> {promotion.ref}")
    if promotion.alias:
        print(
            f"  alias: {promotion.name}@{promotion.alias}"
            + (" (queued for approval)" if promotion.alias_pending else "")
        )
    print(f"  {promotion.copied} artifact(s) copied, including everything behind it")
    if promotion.table:
        print(f"  table: {promotion.table} (snapshot {promotion.table_snapshot})")
    return 0


def cmd_export_table(args: argparse.Namespace) -> int:
    """``strata artifact export``: write a tabular artifact into an Iceberg table."""
    from strata.config import StrataConfig
    from strata.table_export import export_artifact

    store = _open_store(args.artifact_dir)
    if store is None:
        return 2
    artifact = _resolve_for_cmd(store, args)
    if artifact is None:
        return 1
    try:
        written = export_artifact(
            store,
            artifact,
            args.table,
            config=StrataConfig.load(),
            promoted_by=args.by,
            alias=args.alias,
            tenant=getattr(args, "tenant", None),
        )
    except ValueError as exc:
        print(f"Cannot export: {exc}")
        return 1
    if args.format == "json":
        print(
            json.dumps(
                {
                    "artifact_uri": f"strata://artifact/{artifact.id}@v={artifact.version}",
                    "table": written.table,
                    "snapshot_id": written.snapshot_id,
                    "created": written.created,
                },
                indent=2,
            )
        )
        return 0
    verb = "Created" if written.created else "Wrote"
    print(f"{verb} {written.table} from {artifact.id}@v={artifact.version}")
    print(f"  snapshot: {written.snapshot_id}")
    if args.alias:
        print(f"  tag:      {args.alias}")
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    store = _open_store(args.artifact_dir)
    if store is None:
        return 2
    artifact = _resolve_for_cmd(store, args)
    if artifact is None:
        return 1

    target, destination = _publication_target(args, store)
    copied = 0
    published_id, published_version = artifact.id, artifact.version
    if target.db_path != store.db_path:
        written, landed_ref = copy_chain(store, target, artifact, getattr(args, "max_depth", 10))
        copied = len(written)
        # The copy deduplicates against the target, so the grant has to be
        # minted on the row that is actually there. Minting on the source's id
        # would fail on a store that already held the same computation.
        published_id, _, landed_version = landed_ref.partition("@v=")
        published_version = int(landed_version)

    try:
        publication = target.publish_artifact(
            published_id,
            published_version,
            tenant=getattr(args, "tenant", None),
            published_by=getattr(args, "author", None),
            title=args.title,
        )
    except ValueError as exc:
        print(f"Cannot publish: {exc}")
        return 1

    if args.format == "json":
        print(json.dumps(asdict(publication), indent=2))
        return 0

    requested_author = getattr(args, "author", None)
    if requested_author and publication.published_by != requested_author:
        # Publishing is idempotent, so this may have returned an existing grant with its own byline.
        # Say so rather than print success for an author that never reached the page.
        print(
            f"Note: already published, and the page credits "
            f"{publication.published_by or 'nobody'}; republishing does not "
            f"change that. Unpublish and publish again to set an author "
            f"(which mints a new token)."
        )

    print(f"{artifact.id}@v={artifact.version} is public at /p/{publication.token}")
    # Always, not only after a copy: a caller never told where the grant lives cannot tell a working
    # link from one their own server will never resolve.
    print(f"Published into {destination}.")
    if copied:
        plural = "s" if copied != 1 else ""
        print(f"Copied {copied} artifact{plural} across so the link resolves.")
    print()
    print("Anyone with that link can read the artifact, its source, and the")
    print("source and environment of every step behind it. That is the point,")
    print("and it is worth knowing before sending the link:")
    for step in _published_steps(store, artifact, getattr(args, "max_depth", 10)):
        print(f"  - {step}")
    if not getattr(args, "to_url", None):
        # Only a local store can be withdrawn from here, and unpublish reads the configured
        # store unless told which.
        local = getattr(args, "into", None) or (args.artifact_dir if target is store else None)
        command = ["strata", "artifact", "unpublish"]
        if local:
            command += ["--artifact-dir", str(local)]
        if getattr(args, "tenant", None):
            command += ["--tenant", args.tenant]
        print()
        print(f"Withdraw it with: {shlex.join([*command, publication.token])}")
    return 0


def _published_steps(store: ArtifactStore, artifact: ArtifactVersion, max_depth: int) -> list[str]:
    """One line per step whose code and environment the page will expose.

    Deduplicated and normalized to ``id@v=N``: ``_walk_lineage`` renders a tree, so a step
    shared by two cells appears twice. Someone reads this list to decide whether to send a
    link, so it must be the actual set.
    """
    tree = _walk_lineage(store, artifact, max_depth=max_depth)
    steps: list[str] = []

    def _add(label: str) -> None:
        if label and label not in steps:
            steps.append(label)

    def _walk(node: dict) -> None:
        if "artifact_id" in node:
            _add(f"{node['artifact_id']}@v={node['version']}")
        else:
            uri = str(node.get("uri", ""))
            _add(uri.removeprefix("strata://artifact/"))
        for child in node.get("inputs", []):
            _walk(child)

    _walk(tree)
    return steps


def cmd_unpublish(args: argparse.Namespace) -> int:
    from strata.api.publication_bundle import drop_cached_bundles

    store = _open_store(args.artifact_dir)
    if store is None:
        return 2

    if not store.revoke_publication(args.token, tenant=getattr(args, "tenant", None)):
        print("No active publication with that token")
        return 1
    # The route never serves a withdrawn archive, so the cached copy is only disk.
    drop_cached_bundles(store, args.token)

    print("Withdrawn. The link now reports that it was withdrawn rather than")
    print("resolving to anything; it is never reissued for other content.")
    return 0


# --- archive ---


def cmd_archive(args: argparse.Namespace) -> int:
    """Write a self-contained bundle: page, bytes, manifest, README.

    The copy that outlives the server, suitable for a Zenodo or OSF deposit. The bundle is
    :func:`write_bundle` (shared with the HTTP route); this adds the CLI parts: which store to
    open, and whether writing into an occupied directory is a mistake.
    """
    from strata.api.publication_bundle import PUBLICATION_MAX_DEPTH, bundle_zip, write_bundle
    from strata.artifact_store import Publication

    token = getattr(args, "token", None)
    if (token is None) == (args.ref is None):
        print("Give either <ref> or --token, not both.")
        return 1
    if token is not None and (args.title is not None or getattr(args, "author", None)):
        print("--title and --author describe a new record; a publication has its own.")
        return 1

    store = _open_store(args.artifact_dir)
    if store is None:
        return 2

    if token is not None:
        # The stored record, so the bundle carries the authors and identifiers
        # the hosted page shows, and matches GET /p/{token}/archive.zip.
        publication = store.get_publication(token)
        if publication is None:
            print(f"No such publication: {token}")
            return 1
        if not publication.is_active:
            print(f"Publication {token} was withdrawn; its archive is not served.")
            return 1
        artifact = store.get_artifact(publication.artifact_id, publication.version)
        if artifact is None:
            print(f"The artifact {token} publishes is gone.")
            return 1
    else:
        artifact = _resolve_for_cmd(store, args)
        if artifact is None:
            return 1
        # Not inserted in the store: archiving grants nobody access to a running server, so it must
        # not create a publication. The record shape is reused because the page and manifest are the
        # same documents.
        publication = Publication(
            token="",
            artifact_id=artifact.id,
            version=artifact.version,
            title=args.title,
            published_at=time.time(),
            published_by=getattr(args, "author", None),
            content_sha256=store.content_digest(artifact.id, artifact.version),
        )

    max_depth = args.max_depth
    if max_depth is None:
        # By token, the depth the route builds its zip with, so the two stay the same bytes.
        max_depth = PUBLICATION_MAX_DEPTH if token is not None else 10

    dest = Path(args.to)
    if dest.suffix == ".zip":
        if dest.exists() and not getattr(args, "force", False):
            print(f"{dest} exists. Use --force to replace it.")
            return 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(f".{dest.name}.partial")
        try:
            digest = bundle_zip(
                store,
                artifact,
                partial,
                publication=publication,
                max_depth=max_depth,
                tenant=getattr(args, "tenant", None),
            )
        except ValueError as exc:
            print(f"Cannot archive: {exc}")
            return 1
        os.replace(partial, dest)
        print(f"Wrote {dest} (sha256 {digest})")
        return 0

    # A bundle's files describe each other (index.html and README.md name one payload and one
    # digest). Writing into an occupied directory leaves the old payload beside the new one,
    # unnamed, and can clobber a README that was never ours.
    if dest.exists() and any(dest.iterdir()) and not getattr(args, "force", False):
        print(f"{dest}/ is not empty. Use --force to write into it anyway.")
        return 1
    dest.mkdir(parents=True, exist_ok=True)

    try:
        written = write_bundle(
            store,
            artifact,
            dest,
            publication=publication,
            max_depth=max_depth,
            tenant=getattr(args, "tenant", None),
        )
    except ValueError as exc:
        print(f"Cannot archive: {exc}")
        return 1

    print(f"Wrote {dest}/")
    for name in written:
        print(f"  {name}")
    print()
    print("Opens with no server. index.html is the page; manifest.json is the")
    print("same record as machine-readable JSON.")
    return 0


# --- pull ---


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

    blob = store.read_blob(artifact.id, artifact.version)
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


# --- verify ---


def _store_location(store: ArtifactStore) -> tuple[str, str]:
    """Where *store* keeps its metadata and its blobs, printable (no password)."""
    from strata.blob_store import AzureBlobStore, GCSBlobStore, LocalBlobStore, S3BlobStore
    from strata.migrate_cli import _redact
    from strata.sql_backend import PostgresDialect

    dialect = store.dialect
    metadata = _redact(dialect.dsn) if isinstance(dialect, PostgresDialect) else str(store.db_path)
    blob_store = store.blob_store
    if isinstance(blob_store, S3BlobStore):
        blobs = f"s3://{blob_store.bucket}/{blob_store.prefix}"
    elif isinstance(blob_store, GCSBlobStore):
        blobs = f"gs://{blob_store.bucket}/{blob_store.prefix}"
    elif isinstance(blob_store, AzureBlobStore):
        blobs = f"azure://{blob_store.account_name}/{blob_store.container_name}/{blob_store.prefix}"
    elif isinstance(blob_store, LocalBlobStore):
        blobs = str(blob_store.blobs_dir)
    else:
        blobs = type(blob_store).__name__
    return metadata, blobs


def cmd_verify(args: argparse.Namespace) -> int:
    store = _open_store(args.artifact_dir)
    if store is None:
        return 2

    findings = store.verify_artifacts()
    metadata, blobs = _store_location(store)

    if args.format == "json":
        print(
            json.dumps(
                {
                    "artifact_dir": str(store.artifact_dir),
                    "metadata": metadata,
                    "blobs": blobs,
                    "findings": findings,
                },
                indent=2,
            )
        )
    else:
        print(f"verifying: metadata {metadata}, blobs {blobs}")
        if not findings:
            print("\n✓ store is consistent")
        else:
            for f in findings:
                print(f"  ✗ {f['artifact_id']}@v={f['version']} [{f['problem']}] {f['detail']}")
            print(f"\n{len(findings)} problem(s) found")

    return 1 if findings else 0


# --- gc ---

_SIZE_UNITS = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def _parse_size(text: str) -> int:
    """Parse ``20G``, ``500M`` or a plain byte count into bytes.

    Raises:
        ValueError: *text* is not a size.
    """
    cleaned = text.strip().upper().removesuffix("B").removesuffix("I")
    unit = cleaned[-1:] if cleaned[-1:] in _SIZE_UNITS else ""
    number = cleaned[: len(cleaned) - len(unit)]
    try:
        return int(float(number) * _SIZE_UNITS[unit])
    except ValueError:
        raise ValueError(f"not a size: {text!r} (try 20G or 500M)") from None


def cmd_gc(args: argparse.Namespace) -> int:
    from strata.config import StrataConfig

    try:
        policy = StrataConfig.load().artifact_gc_policy()
        if args.max_bytes is not None:
            policy["max_bytes"] = _parse_size(args.max_bytes)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.max_idle_days is not None:
        policy["max_idle_days"] = args.max_idle_days
    if args.min_idle_seconds is not None:
        policy["min_idle_seconds"] = args.min_idle_seconds

    store = _open_store(args.artifact_dir)
    if store is None:
        return 2

    result = store.garbage_collect(
        **policy, collect_latest=args.collect_latest, dry_run=args.dry_run
    )

    if args.format == "json":
        print(json.dumps({"artifact_dir": str(store.artifact_dir), **result}, indent=2))
        return 0

    verb = "would collect" if args.dry_run else "collected"
    print(f"store: {store.artifact_dir} ({_fmt_size(result['store_bytes'])})")
    print(f"{verb} {result['deleted_count']} version(s), {_fmt_size(result['deleted_bytes'])}")
    for entry in result.get("collected", [])[:20]:
        print(
            f"  {entry['artifact_id']}@v={entry['version']}  "
            f"{_fmt_size(entry['byte_size']):>8}  last used {_fmt_when(entry['last_used_at'])}"
        )
    if len(result.get("collected", [])) > 20:
        print(f"  … and {len(result['collected']) - 20} more")
    if policy["max_idle_days"] is None and policy["max_bytes"] is None:
        print("no limit is set (STRATA_ARTIFACT_GC_*, --max-bytes, --max-idle-days): nothing to do")
    return 0


# --- audit / pending ---


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
