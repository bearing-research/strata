"""The tables the ledger owns, for tools that copy or audit a database.

``LEDGER_TABLES`` is in foreign-key order (``artifact_versions`` first) with
each table's primary key, so a row-by-row copy is idempotent. Strata adds its
own tables (API keys, notebook workers) after these when it migrates a store.

Three tables are deliberately absent: ``schema_version`` is created by the
target store itself, and ``import_staging`` and ``build_console_chunks`` hold
only live rows that do not outlive the process that wrote them.
"""

from __future__ import annotations

LEDGER_TABLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("artifact_versions", ("id", "version")),
    ("artifact_names", ("tenant", "name")),
    ("artifact_aliases", ("tenant", "name", "alias")),
    ("artifact_tags", ("tenant", "artifact_id", "version", "key")),
    ("registry_audit", ("seq",)),
    ("registry_pending", ("tenant", "name", "alias")),
    ("artifact_publications", ("token",)),
    ("retention_roots", ("kind", "holder", "artifact_id", "version")),
    (
        "retention_edges",
        ("from_artifact_id", "from_version", "to_artifact_id", "to_version", "kind"),
    ),
    ("artifact_builds", ("build_id",)),
    ("build_attempts", ("build_id", "attempt")),
)

TRANSIENT_TABLES: tuple[str, ...] = ("schema_version", "import_staging", "build_console_chunks")
"""Created by the ledger but never copied between databases."""
