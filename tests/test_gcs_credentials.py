"""``GOOGLE_APPLICATION_CREDENTIALS`` gets a path and nothing else.

Operators paste key material into ``gcs_credentials_json``, often an env var; handed
straight to PyArrow's GCS client it fails only at the first lake read.
"""

import json
import stat
import sys
from pathlib import Path

import pytest

from strata.iceberg import gcs_credentials as module
from strata.iceberg.gcs_credentials import resolve_gcs_credentials


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows has no Unix mode bits; chmod only toggles the read-only flag",
)
def test_inline_key_material_is_written_to_a_private_file():
    key = '{"type": "service_account", "project_id": "p", "private_key": "k"}'

    path = resolve_gcs_credentials(key)

    assert path != key, "inline JSON must become a path"
    written = Path(path)
    assert json.loads(written.read_text()) == json.loads(key)
    # Key material on disk: readable by its owner and nobody else.
    assert stat.S_IMODE(written.stat().st_mode) == 0o600


def test_the_same_key_reuses_one_file():
    # atexit does not run on SIGKILL or an OOM kill, so a random filename
    # would leave one private key behind per hard-killed process. The name
    # is derived from the key, so a restart rewrites the same file.
    key = '{"type": "service_account", "private_key": "k"}'

    first = resolve_gcs_credentials(key)
    second = resolve_gcs_credentials(key)
    other = resolve_gcs_credentials('{"type": "service_account", "private_key": "j"}')

    assert first == second
    assert other != first


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows has no Unix mode bits; chmod only toggles the read-only flag",
)
def test_a_file_left_world_readable_is_tightened():
    # A leftover from a previous run, or a temp dir with a permissive
    # umask: the mode has to be fixed rather than assumed.
    key = '{"type": "service_account", "private_key": "loose"}'
    path = Path(resolve_gcs_credentials(key))
    path.chmod(0o644)

    resolve_gcs_credentials(key)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="no O_NOFOLLOW or file owners there")
def test_a_symlink_planted_at_the_predictable_name_is_not_followed(tmp_path, monkeypatch):
    # The name is derived from the key and visible in the temp dir; another user who
    # saw it plants a link there and the next start would write the key through it.
    monkeypatch.setattr(module.tempfile, "gettempdir", lambda: str(tmp_path))
    key = '{"type": "service_account", "private_key": "SECRET"}'
    target = tmp_path / "theirs.txt"
    target.write_text("")
    planted = Path(resolve_gcs_credentials(key))
    planted.unlink()
    planted.symlink_to(target)

    with pytest.raises(OSError):
        resolve_gcs_credentials(key)

    assert target.read_text() == ""


def test_a_path_is_passed_through_untouched():
    assert resolve_gcs_credentials("/etc/strata/gcs-key.json") == "/etc/strata/gcs-key.json"


def test_a_json_scalar_is_treated_as_a_path():
    # Parsed rather than sniffed for a leading brace, but a bare JSON
    # scalar still parses. A relative filename must not be mistaken for
    # key material just because json.loads accepts it.
    assert resolve_gcs_credentials("123") == "123"
