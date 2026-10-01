"""S3 support: path normalization and joining, S3FileSystem config, metadata extraction.

PyArrow's S3FileSystem bypasses boto3, so moto cannot mock it; test_s3_integration.py uses MinIO.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from strata.config import StrataConfig
from strata.fetcher import PyArrowFetcher, create_fetcher
from strata.metadata_cache import ParquetMetadataCache, reset_caches
from strata.metadata_store import MetadataStore, extract_parquet_meta
from strata.planner import _join_s3_path, _normalize_s3_path
from strata.types import Task


class MockObject:
    """Records attribute access and calls."""

    def __init__(self, **kwargs):
        self._calls = []
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, *args, **kwargs):
        self._calls.append((args, kwargs))
        return self


class TestS3PathNormalization:
    """The S3 path normalization utility."""

    @pytest.mark.parametrize(
        "input_path,expected",
        [
            # Simple paths unchanged
            ("s3://bucket/path/to/file.parquet", "s3://bucket/path/to/file.parquet"),
            # Double slashes collapsed
            ("s3://bucket//path//to//file.parquet", "s3://bucket/path/to/file.parquet"),
            # Trailing slashes removed
            ("s3://bucket/path/to/dir/", "s3://bucket/path/to/dir"),
            # Dot components removed
            ("s3://bucket/./path/./to/./file.parquet", "s3://bucket/path/to/file.parquet"),
            # Double dot navigates up
            ("s3://bucket/path/to/../file.parquet", "s3://bucket/path/file.parquet"),
            # Multiple double dots
            ("s3://bucket/a/b/c/../../file.parquet", "s3://bucket/a/file.parquet"),
            # Bucket only unchanged
            ("s3://bucket", "s3://bucket"),
            # Bucket with trailing slash
            ("s3://bucket/", "s3://bucket"),
            # Complex path with multiple issues
            ("s3://bucket//a/./b/../c//d/./e/../f.parquet", "s3://bucket/a/c/d/f.parquet"),
            # Double dots at start don't go past bucket
            ("s3://bucket/../../../file.parquet", "s3://bucket/file.parquet"),
            # Mixed issues
            ("s3://bucket/./a//b/./c/../d/./e.parquet", "s3://bucket/a/b/d/e.parquet"),
            # Special characters preserved
            (
                "s3://bucket/path/file-with_special.chars.parquet",
                "s3://bucket/path/file-with_special.chars.parquet",
            ),
            (
                "s3://bucket/path/with-dashes_underscores.and.dots/file.parquet",
                "s3://bucket/path/with-dashes_underscores.and.dots/file.parquet",
            ),
        ],
    )
    def test_normalize_s3_path(self, input_path, expected):
        assert _normalize_s3_path(input_path) == expected

    def test_normalize_non_s3_path_unchanged(self):
        path = "/local/path/to/file.parquet"
        assert _normalize_s3_path(path) == path


class TestS3PathJoin:
    """The S3 path joining utility."""

    @pytest.mark.parametrize(
        "base,relative,expected",
        [
            # Simple paths
            (
                "s3://bucket/warehouse",
                "data/file.parquet",
                "s3://bucket/warehouse/data/file.parquet",
            ),
            # Base with trailing slash
            (
                "s3://bucket/warehouse/",
                "data/file.parquet",
                "s3://bucket/warehouse/data/file.parquet",
            ),
            # Relative with leading slash
            (
                "s3://bucket/warehouse",
                "/data/file.parquet",
                "s3://bucket/warehouse/data/file.parquet",
            ),
            # Both trailing and leading slashes
            (
                "s3://bucket/warehouse/",
                "/data/file.parquet",
                "s3://bucket/warehouse/data/file.parquet",
            ),
            # Normalizes result
            (
                "s3://bucket/warehouse",
                "./data/../other/file.parquet",
                "s3://bucket/warehouse/other/file.parquet",
            ),
            # Empty relative
            ("s3://bucket/warehouse", "", "s3://bucket/warehouse"),
            # Bucket only base
            ("s3://bucket", "path/to/file.parquet", "s3://bucket/path/to/file.parquet"),
        ],
    )
    def test_join_s3_path(self, base, relative, expected):
        assert _join_s3_path(base, relative) == expected


@pytest.fixture
def sample_parquet_data():
    return pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5]),
            "name": pa.array(["a", "b", "c", "d", "e"]),
            "value": pa.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        }
    )


@pytest.fixture
def local_parquet_file(tmp_path, sample_parquet_data):
    path = tmp_path / "test.parquet"
    pq.write_table(sample_parquet_data, path)
    return path


class TestS3MetadataExtraction:
    """S3 path handling in metadata extraction."""

    def test_extract_parquet_meta_with_s3_filesystem(self, local_parquet_file):
        """extract_parquet_meta accepts an optional s3_filesystem."""
        mock_fs = MockObject()
        meta = extract_parquet_meta(str(local_parquet_file), s3_filesystem=mock_fs)

        assert meta.num_row_groups >= 1
        assert len(meta.column_names) == 3
        assert "id" in meta.column_names


class TestS3MetadataCache:
    """The metadata cache with S3 paths."""

    def test_cache_loads_local_metadata(self, local_parquet_file):
        reset_caches()
        cache = ParquetMetadataCache(max_size=10)
        metadata = cache.get_or_load(str(local_parquet_file))

        assert metadata is not None
        assert metadata.num_row_groups >= 1
        assert metadata.arrow_schema is not None

    def test_cache_with_s3_filesystem_parameter(self, local_parquet_file):
        reset_caches()
        cache = ParquetMetadataCache(max_size=10, s3_filesystem=MockObject())
        metadata = cache.get_or_load(str(local_parquet_file))

        assert metadata is not None
        assert metadata.num_row_groups >= 1

    def test_cache_persists_metadata(self, tmp_path, sample_parquet_data):
        """Metadata is persisted to the SQLite store."""
        reset_caches()
        pq_path = tmp_path / "test.parquet"
        pq.write_table(sample_parquet_data, pq_path)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        store = MetadataStore(cache_dir / "metadata.sqlite")

        cache = ParquetMetadataCache(max_size=10, store=store)
        cache.get_or_load(str(pq_path))

        persisted = store.get_parquet_meta(str(pq_path))
        assert persisted is not None
        assert persisted.num_row_groups >= 1

    def test_cache_batch_load(self, tmp_path, sample_parquet_data):
        """get_or_load_many works with multiple files."""
        reset_caches()
        paths = []
        for i in range(3):
            path = tmp_path / f"test_{i}.parquet"
            pq.write_table(sample_parquet_data, path)
            paths.append(str(path))

        cache = ParquetMetadataCache(max_size=10)
        results = cache.get_or_load_many(paths)

        assert len(results) == 3
        for path in paths:
            assert path in results
            assert results[path].num_row_groups >= 1


class TestS3Fetcher:
    """Fetcher with S3 paths."""

    @pytest.mark.parametrize(
        "columns,expected_cols",
        [
            (None, 3),
            (["id", "name"], 2),
            (["id"], 1),
        ],
    )
    def test_fetcher_column_projection(self, local_parquet_file, columns, expected_cols):
        fetcher = create_fetcher()
        task = Task(
            file_path=str(local_parquet_file),
            row_group_id=0,
            cache_key=None,  # type: ignore
            num_rows=5,
            columns=columns,
            estimated_bytes=0,
        )

        batch = fetcher.fetch(task)

        assert batch.num_rows == 5
        assert batch.num_columns == expected_cols

    def test_fetcher_accepts_s3_filesystem(self, local_parquet_file):
        fetcher = PyArrowFetcher(s3_filesystem=MockObject())
        task = Task(
            file_path=str(local_parquet_file),
            row_group_id=0,
            cache_key=None,  # type: ignore
            num_rows=5,
            columns=None,
            estimated_bytes=0,
        )

        batch = fetcher.fetch(task)
        assert batch.num_rows == 5

    def test_fetcher_opens_s3_reads_with_the_cached_footer(self, monkeypatch):
        """Later reads reuse the first read's footer via ``metadata=``.

        Every read closes the handle it opened.
        """
        import pyarrow as pa

        import strata.fetcher as fetcher_module
        from strata.types import Task

        created = []

        class FakeParquetFile:
            def __init__(self, path, filesystem=None, metadata=None):
                self.path = path
                self.filesystem = filesystem
                self.given_metadata = metadata
                self.metadata = metadata or object()
                self.closed = False
                created.append(self)

            def read_row_group(self, i, columns=None):
                return pa.table({"id": [1]})

            def close(self):
                self.closed = True

        monkeypatch.setattr(fetcher_module.pq, "ParquetFile", FakeParquetFile)

        fetcher = PyArrowFetcher(max_file_cache_size=4)
        task = Task(file_path="/tmp/one.parquet", row_group_id=0, cache_key=None, num_rows=1)  # type: ignore[arg-type]
        fetcher.fetch(task)
        fetcher.fetch(task)

        assert created[0].given_metadata is None
        assert created[1].given_metadata is created[0].metadata
        assert all(pf.closed for pf in created)

    def test_fetcher_close_forgets_the_cached_footers(self):
        fetcher = PyArrowFetcher(max_file_cache_size=2)
        fetcher._file_cache["/tmp/one.parquet"] = object()  # type: ignore[assignment]
        fetcher.close()
        assert not fetcher._file_cache


class TestS3ConfigIntegration:
    """S3 config integration."""

    @pytest.mark.parametrize(
        "config_kwargs,expected_fs_kwargs",
        [
            # Endpoint only
            (
                {"s3_endpoint_url": "http://localhost:9000"},
                {
                    "endpoint_override": "http://localhost:9000",
                    "connect_timeout": 10.0,
                    "request_timeout": 30.0,
                },
            ),
            # Anonymous
            (
                {"s3_anonymous": True},
                {"anonymous": True, "connect_timeout": 10.0, "request_timeout": 30.0},
            ),
            # All options
            (
                {
                    "s3_region": "us-west-2",
                    "s3_access_key": "access",
                    "s3_secret_key": "secret",
                    "s3_endpoint_url": "http://localhost:9000",
                },
                {
                    "region": "us-west-2",
                    "access_key": "access",
                    "secret_key": "secret",
                    "endpoint_override": "http://localhost:9000",
                    "connect_timeout": 10.0,
                    "request_timeout": 30.0,
                },
            ),
        ],
    )
    def test_config_creates_s3_filesystem(
        self, tmp_path, monkeypatch, config_kwargs, expected_fs_kwargs
    ):
        """Config creates an S3FileSystem with the right options."""
        import pyarrow.fs as pafs

        config = StrataConfig(cache_dir=tmp_path, **config_kwargs)

        s3_fs_calls = []

        def mock_s3_filesystem(**kwargs):
            s3_fs_calls.append(kwargs)
            return MockObject()

        monkeypatch.setattr(pafs, "S3FileSystem", mock_s3_filesystem)
        config.get_s3_filesystem()

        assert len(s3_fs_calls) == 1
        assert s3_fs_calls[0] == expected_fs_kwargs

    @pytest.mark.parametrize(
        "has_s3_config,expected_calls",
        [
            (True, 1),
            (False, 0),
        ],
    )
    def test_planner_s3_fs_creation(self, tmp_path, monkeypatch, has_s3_config, expected_calls):
        """ReadPlanner creates an S3FileSystem only when S3 config is present."""
        config_kwargs = {"cache_dir": tmp_path}
        if has_s3_config:
            config_kwargs["s3_endpoint_url"] = "http://minio:9000"

        config = StrataConfig(**config_kwargs)

        get_fs_calls = []

        def mock_get_fs(self):
            get_fs_calls.append(True)
            return MockObject()

        # Patch method on the class instead of instance (Pydantic models are frozen)
        monkeypatch.setattr(StrataConfig, "get_s3_filesystem", mock_get_fs)

        from strata.planner import ReadPlanner

        ReadPlanner(config)

        assert len(get_fs_calls) == expected_calls


class TestS3EdgeCases:
    def test_empty_parquet_file(self, tmp_path):
        empty_table = pa.table({"id": pa.array([], type=pa.int64())})
        path = tmp_path / "empty.parquet"
        pq.write_table(empty_table, path)

        meta = extract_parquet_meta(str(path))

        assert meta.num_row_groups >= 0
        assert "id" in meta.column_names

    def test_file_prefix_stripped(self, tmp_path, sample_parquet_data):
        """The planner handles the file:// prefix."""
        from strata.planner import ReadPlanner

        pq_path = tmp_path / "test.parquet"
        pq.write_table(sample_parquet_data, pq_path)

        planner = ReadPlanner.__new__(ReadPlanner)
        file_path = f"file://{pq_path}"
        resolved = planner._resolve_file_path("unused#ns.table", file_path)

        assert resolved == str(pq_path)
        assert not resolved.startswith("file://")
