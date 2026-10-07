"""``strata apikey``: mint, list, and revoke API keys.

Talks to the key store directly and needs no running server, which is what
makes bootstrapping the first key possible.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from strata.api_keys import ApiKeyStore
from strata.file_modes import private_dir, private_file

_SECONDS_PER_DAY = 86400.0


def _open_store(args: argparse.Namespace) -> ApiKeyStore:
    """Open the key store the server reads.

    ``--artifact-dir`` names one local SQLite store. Without it, the server's configured
    store (``[tool.strata]``, then ``STRATA_*``): its ``artifact_dir`` and metadata DSN, so
    keys never land in a file the server ignores. ``--dsn`` overrides either.
    """
    if args.artifact_dir:
        artifact_dir = Path(args.artifact_dir)
        dialect = None
    else:
        from strata.config import StrataConfig

        try:
            config = StrataConfig.load()
            dialect = None if args.dsn else config.create_metadata_dialect()
        except ValueError as exc:
            raise SystemExit(f"invalid configuration: {exc}") from exc
        artifact_dir = config.artifact_dir or Path.home() / ".strata" / "artifacts"
    if args.dsn:
        from strata.sql_backend import PostgresDialect

        dialect = PostgresDialect(args.dsn)

    # Often run before the first server start, so this may create the store the server opens.
    private_dir(artifact_dir)
    if dialect is None:
        private_file(artifact_dir / "artifacts.sqlite")
    return ApiKeyStore(artifact_dir / "artifacts.sqlite", dialect=dialect)


def _format_time(value: float | None) -> str:
    if value is None:
        return "-"
    return datetime.fromtimestamp(value, UTC).strftime("%Y-%m-%d %H:%M:%SZ")


def cmd_create(args: argparse.Namespace) -> int:
    store = _open_store(args)
    expires_in = args.expires_in_days * _SECONDS_PER_DAY if args.expires_in_days else None

    presented, record = store.create_key(
        principal_id=args.principal,
        tenant=args.tenant,
        scopes=frozenset(args.scopes or ()),
        description=args.description,
        expires_in_seconds=expires_in,
    )

    # Only the key goes to stdout, so ``KEY=$(strata apikey create ...)``
    # captures the credential and nothing else.
    print(presented)

    summary = [
        "",
        f"  key id     {record.key_id}",
        f"  principal  {record.principal_id}",
        f"  tenant     {record.tenant or '-'}",
        f"  scopes     {' '.join(sorted(record.scopes)) or '-'}",
        f"  expires    {_format_time(record.expires_at)}",
        "",
        "Store it now. Only the hash is kept, so it cannot be shown again.",
    ]
    print("\n".join(summary), file=sys.stderr)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    store = _open_store(args)
    records = store.list_keys(principal_id=args.principal)

    if args.format == "json":
        print(
            json.dumps(
                [
                    {
                        "key_id": r.key_id,
                        "principal_id": r.principal_id,
                        "tenant": r.tenant,
                        "scopes": sorted(r.scopes),
                        "description": r.description,
                        "created_at": r.created_at,
                        "expires_at": r.expires_at,
                        "revoked_at": r.revoked_at,
                        "last_used_at": r.last_used_at,
                        "active": r.is_active,
                    }
                    for r in records
                ],
                indent=2,
            )
        )
        return 0

    if not records:
        print("No API keys.")
        return 0

    print(f"{'KEY ID':<34} {'PRINCIPAL':<20} {'STATUS':<9} {'LAST USED':<21} DESCRIPTION")
    for r in records:
        status = "active" if r.is_active else ("revoked" if r.revoked_at else "expired")
        print(
            f"{r.key_id:<34} {r.principal_id:<20} {status:<9} "
            f"{_format_time(r.last_used_at):<21} {r.description or '-'}"
        )
    return 0


def cmd_revoke(args: argparse.Namespace) -> int:
    store = _open_store(args)
    if store.revoke(args.key_id):
        print(f"Revoked {args.key_id}.")
        return 0
    # Already revoked, or never existed. Both mean "not live", and the
    # distinction is not worth a different exit code to a script.
    print(f"No live key {args.key_id}.")
    return 1
