"""The ledger of the shared computation substrate (ADR 002); *vestigo*, I trace.

``store`` holds versioned artifacts, names, aliases, tags, pins, publications,
retention and the audit log; ``build_store`` the builds, their leases and
attempts; ``releases`` a build's checkpoint and final releases, fenced on its
attempt's lease, with ``operations`` the typed operations a release runs in its
transaction; ``graph`` the content-addressed structural nodes, their encoders
and the bounded traversals; ``sql_backend`` the SQLite and PostgreSQL dialects
and the one write discipline they share; ``blob_store`` the byte backends;
``provenance`` the transform spec, its hash and the per-output key derivation;
``uris`` the artifact and name URI grammar and the process's scheme;
``file_modes`` the private directory and file modes the stores create with;
``tables`` the list of tables the ledger owns.
"""
