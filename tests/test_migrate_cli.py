"""``strata migrate``'s refusals, which need no Postgres: each exits 1 and says why.

The runs against a real target are in ``test_migrate_to_postgres.py``.
"""

from __future__ import annotations

from vestigo.store import ArtifactStore

from strata.cli import main


def test_a_directory_without_a_store_is_refused(tmp_path, capsys):
    assert main(["migrate", "--artifact-dir", str(tmp_path), "--to-dsn", "postgresql://h/db"]) == 1

    assert f"No SQLite store at {tmp_path / 'artifacts.sqlite'}" in capsys.readouterr().out


def test_no_target_is_refused(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("STRATA_ARTIFACT_METADATA_DSN", raising=False)
    ArtifactStore(tmp_path)

    assert main(["migrate", "--artifact-dir", str(tmp_path)]) == 1

    assert (
        "No target. Pass --to-dsn or set STRATA_ARTIFACT_METADATA_DSN." in capsys.readouterr().out
    )


def test_a_target_that_is_not_postgres_is_refused_before_connecting(tmp_path, capsys):
    ArtifactStore(tmp_path)

    assert main(["migrate", "--artifact-dir", str(tmp_path), "--to-dsn", "mysql://h/db"]) == 1

    out = capsys.readouterr().out
    assert out.startswith("Refused: artifact_metadata_dsn must be a postgresql:// URL")
