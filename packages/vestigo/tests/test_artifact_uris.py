"""Unit tests for the pure artifact/name URI parsers in ``vestigo.uris``."""

import pytest

from vestigo import uris
from vestigo.uris import (
    LATEST_VERSION,
    parse_artifact_uri,
    parse_name_uri,
)


class TestParseArtifactUri:
    def test_pinned_version(self):
        assert parse_artifact_uri("vestigo://artifact/abc@v=3") == ("abc", 3)

    def test_pinned_version_multi_digit(self):
        assert parse_artifact_uri("vestigo://artifact/abc@v=142") == ("abc", 142)

    def test_latest_when_unpinned(self):
        assert parse_artifact_uri("vestigo://artifact/abc") == ("abc", LATEST_VERSION)

    def test_id_with_underscores_and_dashes(self):
        assert parse_artifact_uri("vestigo://artifact/nb_42_cell_x-1@v=7") == (
            "nb_42_cell_x-1",
            7,
        )

    def test_at_stops_id_capture(self):
        # ``[^@]+`` must not swallow the ``@v=`` suffix into the id.
        artifact_id, version = parse_artifact_uri("vestigo://artifact/xyz@v=9")
        assert artifact_id == "xyz"
        assert version == 9

    def test_name_uri_is_not_an_artifact_uri(self):
        assert parse_artifact_uri("vestigo://name/foo") is None

    def test_another_scheme_is_not_one(self):
        assert parse_artifact_uri("http://example.com/foo") is None

    def test_non_numeric_version_is_not_pinned(self):
        # ``@v=x`` matches neither the pinned nor the latest pattern.
        assert parse_artifact_uri("vestigo://artifact/abc@v=x") is None

    def test_empty_string(self):
        assert parse_artifact_uri("") is None


class TestParseNameUri:
    def test_basic_name(self):
        assert parse_name_uri("vestigo://name/my-result") == "my-result"

    def test_name_with_slashes(self):
        # ``.+`` keeps everything after the prefix, including path-like names.
        assert parse_name_uri("vestigo://name/team/report") == "team/report"

    def test_artifact_uri_is_not_a_name_uri(self):
        assert parse_name_uri("vestigo://artifact/abc@v=1") is None

    def test_empty_name_rejected(self):
        assert parse_name_uri("vestigo://name/") is None

    def test_another_scheme_is_not_one(self):
        assert parse_name_uri("s3://bucket/key") is None


class TestUriScheme:
    """The scheme is the consumer's: set once per process, it reaches every parse and format."""

    @pytest.fixture(autouse=True)
    def _default_scheme_after(self):
        yield
        uris.set_uri_scheme(uris.DEFAULT_URI_SCHEME)

    def test_the_default_is_vestigo(self):
        assert uris.uri_scheme() == "vestigo"
        assert uris.artifact_uri("abc", 3) == "vestigo://artifact/abc@v=3"
        assert uris.artifact_uri_prefix() == "vestigo://artifact/"
        assert uris.name_uri_prefix() == "vestigo://name/"

    def test_a_consumer_scheme_is_parsed_and_written(self):
        uris.set_uri_scheme("strata")
        assert uris.artifact_uri("abc", 3) == "strata://artifact/abc@v=3"
        assert parse_artifact_uri("strata://artifact/abc@v=3") == ("abc", 3)
        assert parse_artifact_uri("strata://artifact/abc") == ("abc", LATEST_VERSION)
        assert parse_name_uri("strata://name/taxi/model@champion") == "taxi/model@champion"
        # The default scheme is now foreign.
        assert parse_artifact_uri("vestigo://artifact/abc@v=3") is None
        assert parse_name_uri("vestigo://name/x") is None

    @pytest.mark.parametrize("scheme", ["", "1abc", "my scheme", "Strata", "a://"])
    def test_a_name_that_is_not_a_scheme_is_refused(self, scheme):
        with pytest.raises(ValueError, match="not a URI scheme"):
            uris.set_uri_scheme(scheme)
        assert uris.uri_scheme() == "vestigo"
