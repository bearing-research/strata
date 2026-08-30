"""``strata migrate`` — move an existing SQLite store onto Postgres."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from strata.migrate import migrate, plan_migration
from strata.sql_backend import PostgresDialect, SqliteDialect


def cmd_migrate(args: argparse.Namespace) -> int:
    artifact_dir = Path(args.artifact_dir or Path.home() / ".strata" / "artifacts")
    source_db = artifact_dir / "artifacts.sqlite"
    if not source_db.exists():
        print(f"No SQLite store at {source_db}.")
        return 1

    dsn = args.to_dsn or os.environ.get("STRATA_ARTIFACT_METADATA_DSN")
    if not dsn:
        print("No target. Pass --to-dsn or set STRATA_ARTIFACT_METADATA_DSN.")
        return 1

    source = SqliteDialect(source_db)
    target = PostgresDialect(dsn)
    try:
        plan = plan_migration(source, target)

        print(f"Source  {source_db}")
        print(f"Target  {_redact(dsn)}")
        print()
        print(f"{'TABLE':<22} {'SOURCE':>8} {'TARGET':>8}")
        for table in plan.source_counts:
            print(f"{table:<22} {plan.source_counts[table]:>8} {plan.target_counts[table]:>8}")
        print()

        if plan.missing_in_target:
            # The stores create their own schema at startup; deriving the DDL
            # here would give it a second definition free to drift.
            print(f"Target is missing: {', '.join(plan.missing_in_target)}")
            print("Start Strata against the target database once, then migrate.")
            return 1

        if args.dry_run:
            print(f"Dry run: {plan.total_rows} rows would be considered.")
            return 0

        result = migrate(source, target, allow_nonempty_target=args.allow_nonempty_target)
        print(f"Copied {result.total_copied} rows, skipped {result.total_skipped} already present.")
        print()
        print("Blobs are not copied. Point the target deployment at the same blob")
        print("backend, or the metadata will resolve to bytes it cannot read.")
        return 0
    except ValueError as exc:
        print(f"Refused: {exc}")
        return 1
    finally:
        target.close()


def _redact(dsn: str) -> str:
    """Hide the password so a migration transcript is safe to paste."""
    if "@" not in dsn or "://" not in dsn:
        return dsn
    scheme, rest = dsn.split("://", 1)
    credentials, host = rest.rsplit("@", 1)
    user = credentials.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"
