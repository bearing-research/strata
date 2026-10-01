"""Tests for S3 storage backend configuration."""

from unittest.mock import MagicMock, patch

from strata.config import StrataConfig


class TestS3ConfigDefaults:
    """Test S3 configuration defaults."""

    def test_default_s3_settings_are_none(self):
        """S3 settings default to None/False."""
        config = StrataConfig()
        assert config.s3_region is None
        assert config.s3_access_key is None
        assert config.s3_secret_key is None
        assert config.s3_endpoint_url is None
        assert config.s3_anonymous is False

    def test_s3_config_can_be_set_directly(self, tmp_path):
        """S3 settings can be set programmatically."""
        config = StrataConfig(
            cache_dir=tmp_path,
            s3_region="us-west-2",
            s3_access_key="test-key",
            s3_secret_key="test-secret",
            s3_endpoint_url="http://localhost:9000",
            s3_anonymous=False,
        )
        assert config.s3_region == "us-west-2"
        assert config.s3_access_key == "test-key"
        assert config.s3_secret_key == "test-secret"
        assert config.s3_endpoint_url == "http://localhost:9000"
        assert config.s3_anonymous is False


class TestS3EnvironmentOverrides:
    """Test S3 configuration from environment variables."""

    def test_strata_s3_region_env(self, tmp_path, monkeypatch):
        """STRATA_S3_REGION environment variable is read."""
        monkeypatch.setenv("STRATA_S3_REGION", "eu-west-1")
        config = StrataConfig.load(cache_dir=tmp_path)
        assert config.s3_region == "eu-west-1"

    def test_aws_region_env_fallback(self, tmp_path, monkeypatch):
        """AWS_REGION is used as fallback for S3 region."""
        monkeypatch.setenv("AWS_REGION", "ap-south-1")
        config = StrataConfig.load(cache_dir=tmp_path)
        assert config.s3_region == "ap-south-1"

    def test_strata_s3_region_takes_precedence(self, tmp_path, monkeypatch):
        """STRATA_S3_REGION takes precedence over AWS_REGION."""
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setenv("STRATA_S3_REGION", "us-west-2")
        config = StrataConfig.load(cache_dir=tmp_path)
        assert config.s3_region == "us-west-2"

    def test_aws_access_key_env(self, tmp_path, monkeypatch):
        """AWS_ACCESS_KEY_ID environment variable is read."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "my-access-key")
        config = StrataConfig.load(cache_dir=tmp_path)
        assert config.s3_access_key == "my-access-key"

    def test_strata_s3_access_key_takes_precedence(self, tmp_path, monkeypatch):
        """STRATA_S3_ACCESS_KEY takes precedence over AWS_ACCESS_KEY_ID."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "aws-key")
        monkeypatch.setenv("STRATA_S3_ACCESS_KEY", "strata-key")
        config = StrataConfig.load(cache_dir=tmp_path)
        assert config.s3_access_key == "strata-key"

    def test_aws_secret_key_env(self, tmp_path, monkeypatch):
        """AWS_SECRET_ACCESS_KEY environment variable is read."""
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "my-secret")
        config = StrataConfig.load(cache_dir=tmp_path)
        assert config.s3_secret_key == "my-secret"

    def test_s3_endpoint_url_env(self, tmp_path, monkeypatch):
        """STRATA_S3_ENDPOINT_URL environment variable is read."""
        monkeypatch.setenv("STRATA_S3_ENDPOINT_URL", "http://minio:9000")
        config = StrataConfig.load(cache_dir=tmp_path)
        assert config.s3_endpoint_url == "http://minio:9000"

    def test_s3_anonymous_env(self, tmp_path, monkeypatch):
        """STRATA_S3_ANONYMOUS environment variable is read."""
        monkeypatch.setenv("STRATA_S3_ANONYMOUS", "true")
        config = StrataConfig.load(cache_dir=tmp_path)
        assert config.s3_anonymous is True

    def test_s3_anonymous_false(self, tmp_path, monkeypatch):
        """STRATA_S3_ANONYMOUS=false is handled correctly."""
        monkeypatch.setenv("STRATA_S3_ANONYMOUS", "false")
        config = StrataConfig.load(cache_dir=tmp_path)
        assert config.s3_anonymous is False


class TestS3FilesystemFactory:
    """Test S3 filesystem creation from config."""

    def test_get_s3_filesystem_with_region(self, tmp_path):
        """S3FileSystem is created with region."""
        config = StrataConfig(
            cache_dir=tmp_path,
            s3_region="us-east-1",
        )

        with patch("pyarrow.fs.S3FileSystem") as mock_s3:
            mock_s3.return_value = MagicMock()
            config.get_s3_filesystem()
            mock_s3.assert_called_once_with(
                region="us-east-1",
                connect_timeout=10.0,
                request_timeout=30.0,
            )

    def test_get_s3_filesystem_with_credentials(self, tmp_path):
        """S3FileSystem is created with credentials."""
        config = StrataConfig(
            cache_dir=tmp_path,
            s3_access_key="my-key",
            s3_secret_key="my-secret",
        )

        with patch("pyarrow.fs.S3FileSystem") as mock_s3:
            mock_s3.return_value = MagicMock()
            config.get_s3_filesystem()
            mock_s3.assert_called_once_with(
                access_key="my-key",
                secret_key="my-secret",
                connect_timeout=10.0,
                request_timeout=30.0,
            )

    def test_get_s3_filesystem_with_endpoint(self, tmp_path):
        """S3FileSystem is created with custom endpoint."""
        config = StrataConfig(
            cache_dir=tmp_path,
            s3_endpoint_url="http://localhost:9000",
        )

        with patch("pyarrow.fs.S3FileSystem") as mock_s3:
            mock_s3.return_value = MagicMock()
            config.get_s3_filesystem()
            mock_s3.assert_called_once_with(
                endpoint_override="http://localhost:9000",
                connect_timeout=10.0,
                request_timeout=30.0,
            )

    def test_get_s3_filesystem_with_anonymous(self, tmp_path):
        """S3FileSystem is created with anonymous access."""
        config = StrataConfig(
            cache_dir=tmp_path,
            s3_anonymous=True,
        )

        with patch("pyarrow.fs.S3FileSystem") as mock_s3:
            mock_s3.return_value = MagicMock()
            config.get_s3_filesystem()
            mock_s3.assert_called_once_with(
                anonymous=True,
                connect_timeout=10.0,
                request_timeout=30.0,
            )

    def test_get_s3_filesystem_all_options(self, tmp_path):
        """S3FileSystem is created with all options."""
        config = StrataConfig(
            cache_dir=tmp_path,
            s3_region="us-west-2",
            s3_access_key="key",
            s3_secret_key="secret",
            s3_endpoint_url="http://minio:9000",
        )

        with patch("pyarrow.fs.S3FileSystem") as mock_s3:
            mock_s3.return_value = MagicMock()
            config.get_s3_filesystem()
            mock_s3.assert_called_once_with(
                region="us-west-2",
                access_key="key",
                secret_key="secret",
                endpoint_override="http://minio:9000",
                connect_timeout=10.0,
                request_timeout=30.0,
            )


class TestS3URIParsing:
    """Test S3 URI parsing in iceberg module."""

    def test_parse_s3_table_uri(self):
        """S3 table URIs are parsed correctly."""
        from strata.iceberg import PyIcebergCatalog

        warehouse, table_id = PyIcebergCatalog.parse_table_uri("s3://my-bucket/warehouse#db.table")
        assert warehouse == "s3://my-bucket/warehouse"
        assert table_id == "db.table"

    def test_parse_s3_uri_preserves_prefix(self):
        """S3 prefix is preserved (not stripped like file://)."""
        from strata.iceberg import PyIcebergCatalog

        warehouse, table_id = PyIcebergCatalog.parse_table_uri(
            "s3://bucket/path/to/warehouse#namespace.table"
        )
        assert warehouse is not None
        assert warehouse.startswith("s3://")
        assert warehouse == "s3://bucket/path/to/warehouse"

    def test_parse_local_uri_strips_file_prefix(self):
        """Local file:// URIs still strip the prefix."""
        from strata.iceberg import PyIcebergCatalog

        warehouse, table_id = PyIcebergCatalog.parse_table_uri("file:///path/to/warehouse#db.table")
        assert warehouse == "/path/to/warehouse"
        assert not warehouse.startswith("file://")


class TestS3PathResolution:
    """Test S3 path resolution in planner."""

    def test_resolve_s3_absolute_path(self, tmp_path):
        """Absolute S3 paths are preserved."""
        from strata.config import StrataConfig
        from strata.planner import ReadPlanner

        config = StrataConfig(cache_dir=tmp_path)
        planner = ReadPlanner(config)

        resolved = planner._resolve_file_path(
            "s3://bucket/warehouse#db.table",
            "s3://bucket/warehouse/data/file.parquet",
        )
        assert resolved == "s3://bucket/warehouse/data/file.parquet"

    def test_resolve_s3_relative_path(self, tmp_path):
        """Relative paths in S3 warehouse are resolved."""
        from strata.config import StrataConfig
        from strata.planner import ReadPlanner

        config = StrataConfig(cache_dir=tmp_path)
        planner = ReadPlanner(config)

        resolved = planner._resolve_file_path(
            "s3://bucket/warehouse#db.table",
            "data/file.parquet",
        )
        assert resolved == "s3://bucket/warehouse/data/file.parquet"
