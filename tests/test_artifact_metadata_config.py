"""Selecting the artifact store's metadata backend from configuration.

The Postgres backend was reachable only from direct instantiation until this
landed, so these cover the wiring: the DSN field, the dialect factory, the
startup validation, and the coherence rule that keeps a shared database from
being paired with node-local blobs.
"""

from __future__ import annotations

import pytest

from strata.config import StrataConfig


class TestMetadataDialectFactory:
    def test_unset_dsn_keeps_sqlite(self):
        # Every existing deployment: no DSN, no dialect, SQLite under
        # artifact_dir exactly as before.
        assert StrataConfig().create_metadata_dialect() is None

    def test_postgres_dsn_builds_a_postgres_dialect(self):
        dialect = StrataConfig(
            artifact_metadata_dsn="postgresql://u:p@db:5432/strata"
        ).create_metadata_dialect()
        assert dialect is not None
        assert dialect.name == "postgres"

    def test_postgres_scheme_alias_is_accepted(self):
        # libpq accepts both spellings and operators copy whichever their
        # provider prints.
        dialect = StrataConfig(
            artifact_metadata_dsn="postgres://u:p@db:5432/strata"
        ).create_metadata_dialect()
        assert dialect is not None

    def test_unsupported_scheme_is_rejected_at_startup(self):
        # Failing here beats failing on the store's first write, which would
        # mean work was accepted that could not be kept.
        with pytest.raises(ValueError, match="must be a postgresql"):
            StrataConfig(artifact_metadata_dsn="mysql://u:p@db/strata").create_metadata_dialect()

    def test_a_bare_path_is_rejected(self):
        with pytest.raises(ValueError, match="must be a postgresql"):
            StrataConfig(artifact_metadata_dsn="/var/lib/strata.db").create_metadata_dialect()

    def test_building_the_dialect_opens_no_connection(self):
        # The DSN points at nothing here. Construction must stay lazy, or an
        # unreachable database would make the server fail to boot even when
        # nothing has queried yet.
        dialect = StrataConfig(
            artifact_metadata_dsn="postgresql://nobody@127.0.0.1:1/none"
        ).create_metadata_dialect()
        assert dialect._pool is None


class TestSharedMetadataRequiresSharedBlobs:
    """A shared database with node-local blobs is only coherent on one machine."""

    def _service(self, **kwargs):
        return dict(
            deployment_mode="service",
            artifact_dir="/tmp/strata-artifacts",
            auth_mode="trusted_proxy",
            proxy_token="t",
            **kwargs,
        )

    def test_dsn_with_local_blobs_is_rejected(self):
        # Node B resolves an artifact from the shared database, then looks for
        # bytes that only exist on node A's disk.
        with pytest.raises(ValueError, match="blobs are not"):
            StrataConfig(
                **self._service(
                    artifact_metadata_dsn="postgresql://u:p@db/strata",
                    artifact_blob_backend="local",
                )
            )

    def test_dsn_with_shared_blobs_is_accepted(self):
        config = StrataConfig(
            **self._service(
                artifact_metadata_dsn="postgresql://u:p@db/strata",
                artifact_blob_backend="s3",
                artifact_s3_bucket="strata-artifacts",
            )
        )
        assert config.artifact_metadata_dsn is not None

    def test_local_blobs_alone_stay_fine(self):
        # The rule is about the *combination*; local blobs without a shared
        # database is the ordinary single-node deployment.
        config = StrataConfig(**self._service(artifact_blob_backend="local"))
        assert config.artifact_metadata_dsn is None


class TestEnvironmentVariable:
    def test_dsn_is_settable_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("STRATA_ARTIFACT_METADATA_DSN", "postgresql://u:p@db/strata")
        assert StrataConfig().artifact_metadata_dsn == "postgresql://u:p@db/strata"
