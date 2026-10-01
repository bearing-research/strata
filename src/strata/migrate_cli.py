"""``strata migrate`` — move an existing SQLite store onto Postgres."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from strata.migrate import migrate, plan_migration
from strata.sql_backend import SqliteDialect


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

    # Validate the DSN as the server does, so an unsupported scheme or missing
    # [postgres] extra gives the usual message, not a driver traceback.
    from strata.config import StrataConfig

    try:
        target = StrataConfig(artifact_metadata_dsn=dsn).create_metadata_dialect()
    except ValueError as exc:
        print(f"Refused: {exc}")
        return 1
    if target is None:  # pragma: no cover - guarded by the check above
        print("Refused: no target dialect for that DSN.")
        return 1

    source = SqliteDialect(source_db)
    try:
        plan = plan_migration(source, target)

        print(f"Source  {source_db}")
        print(f"Target  {_redact(dsn)}")
        print()
        print(f"{'TABLE':<22} {'SOURCE':>8} {'TARGET':>8}")
        for table in plan.source_counts:
            print(f"{table:<22} {plan.source_counts[table]:>8} {plan.target_counts[table]:>8}")
        print()

        if plan.blocking_tables:
            # Only tables the source has rows for: api_keys and artifact_builds
            # are legitimately absent on a target booted the documented way.
            # The stores create their own schema; DDL derived here would drift.
            print(f"Target is missing: {', '.join(plan.blocking_tables)}")
            print("Start Strata against the target database once, then migrate.")
            return 1

        if args.dry_run:
            print(f"Dry run: {plan.total_rows} rows would be considered.")
            if not plan.target_is_empty:
                # Reported, not failed: the operator may intend to resume.
                print("Target already holds rows; the real run needs --allow-nonempty-target.")
            return 0

        # Only for a real run: ArtifactStore's schema init upgrades a legacy
        # source in place (columns, indexes, WAL), which --dry-run must not do.
        from strata.artifact_store import ArtifactStore

        ArtifactStore(artifact_dir)

        result = migrate(source, target, allow_nonempty_target=args.allow_nonempty_target)
        print(f"Copied {result.total_copied} rows, skipped {result.total_skipped} already present.")

        if result.rejected:
            # Usually build rows whose artifact version was garbage-collected or
            # deleted: dangling references SQLite tolerated and Postgres does not.
            print()
            print(f"{result.total_rejected} rows were refused by the target:")
            for table, key, reason in result.rejected[:10]:
                print(f"  {table} {key}: {reason}")
            if result.total_rejected > 10:
                print(f"  ... and {result.total_rejected - 10} more")
        print()
        print("Blobs are not copied. Point the target deployment at the same blob")
        print("backend, or the metadata will resolve to bytes it cannot read.")

        # Non-zero when anything was refused, so `strata migrate && cut-over`
        # does not switch traffic to a target that is missing rows.
        return 1 if result.rejected else 0
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
