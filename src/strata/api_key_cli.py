"""``strata apikey`` — mint, list, and revoke API keys.

Ships alongside the key store rather than after it, because a credential
system with no way to issue a credential is not usable, only present. The
commands talk to the store directly and need no running server, which is what
makes bootstrapping the first key possible.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from strata.api_keys import ApiKeyStore

_SECONDS_PER_DAY = 86400.0


def _open_store(args: argparse.Namespace) -> ApiKeyStore:
    """Open the key store the same way the server would.

    The DSN precedence matches ``StrataConfig``: an explicit ``--dsn`` first,
    then the environment, then SQLite under the artifact directory. An operator
    who has moved the metadata to Postgres must not have the CLI quietly mint
    keys into a local file the server never reads.
    """
    artifact_dir = Path(args.artifact_dir or Path.home() / ".strata" / "artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    dsn = args.dsn or os.environ.get("STRATA_ARTIFACT_METADATA_DSN")
    dialect = None
    if dsn:
        from strata.sql_backend import PostgresDialect

        dialect = PostgresDialect(dsn)

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

    print(presented)
    print()
    print(f"  key id     {record.key_id}")
    print(f"  principal  {record.principal_id}")
    print(f"  tenant     {record.tenant or '-'}")
    print(f"  scopes     {' '.join(sorted(record.scopes)) or '-'}")
    print(f"  expires    {_format_time(record.expires_at)}")
    print()
    print("Store it now. Only the hash is kept, so it cannot be shown again.")
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
