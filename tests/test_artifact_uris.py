"""Unit tests for the pure artifact/name URI parsers.

These parsers were previously private helpers inside ``server.py`` exercised
only through the ``/v1/materialize`` endpoints; extracting them to
``strata.artifact_uris`` lets us pin their grammar directly.
"""

from strata.artifact_uris import (
    LATEST_VERSION,
    parse_artifact_uri,
    parse_name_uri,
)


class TestParseArtifactUri:
    def test_pinned_version(self):
        assert parse_artifact_uri("strata://artifact/abc@v=3") == ("abc", 3)

    def test_pinned_version_multi_digit(self):
        assert parse_artifact_uri("strata://artifact/abc@v=142") == ("abc", 142)

    def test_latest_when_unpinned(self):
        assert parse_artifact_uri("strata://artifact/abc") == ("abc", LATEST_VERSION)

    def test_id_with_underscores_and_dashes(self):
        assert parse_artifact_uri("strata://artifact/nb_42_cell_x-1@v=7") == (
            "nb_42_cell_x-1",
            7,
        )

    def test_at_stops_id_capture(self):
        # ``[^@]+`` must not swallow the ``@v=`` suffix into the id.
        artifact_id, version = parse_artifact_uri("strata://artifact/xyz@v=9")
        assert artifact_id == "xyz"
        assert version == 9

    def test_name_uri_is_not_an_artifact_uri(self):
        assert parse_artifact_uri("strata://name/foo") is None

    def test_non_strata_uri(self):
        assert parse_artifact_uri("http://example.com/foo") is None

    def test_non_numeric_version_is_not_pinned(self):
        # ``@v=x`` matches neither the pinned nor the latest pattern.
        assert parse_artifact_uri("strata://artifact/abc@v=x") is None

    def test_empty_string(self):
        assert parse_artifact_uri("") is None


class TestParseNameUri:
    def test_basic_name(self):
        assert parse_name_uri("strata://name/my-result") == "my-result"

    def test_name_with_slashes(self):
        # ``.+`` keeps everything after the prefix, including path-like names.
        assert parse_name_uri("strata://name/team/report") == "team/report"

    def test_artifact_uri_is_not_a_name_uri(self):
        assert parse_name_uri("strata://artifact/abc@v=1") is None

    def test_empty_name_rejected(self):
        assert parse_name_uri("strata://name/") is None

    def test_non_strata_uri(self):
        assert parse_name_uri("s3://bucket/key") is None
