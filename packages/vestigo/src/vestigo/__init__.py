"""The ledger of the shared computation substrate (ADR 002); *vestigo*, I trace.

``store`` holds versioned artifacts, names, aliases, tags, pins, publications
and the audit log; ``build_store`` the builds, their leases and attempts;
``sql_backend`` the SQLite and PostgreSQL dialects and the one write
discipline they share; ``blob_store`` the byte backends; ``provenance`` the
transform spec, its hash and the per-output key derivation; ``uris`` the
artifact URI grammar; ``tables`` the list of tables the ledger owns.
"""
