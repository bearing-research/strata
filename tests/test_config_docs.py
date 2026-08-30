"""The configuration reference and the config surface must not drift apart.

Both directions are failures an operator pays for:

- A documented setting that does not exist. ``StrataConfig`` sets
  ``extra="ignore"``, so an unknown ``STRATA_*`` is accepted in silence -- the
  operator sets it, sees no error, and believes something is configured.
  ``STRATA_PULL_MODEL_ENABLED`` was documented as gating the signed-URL routes
  long after the flag was deleted; the routes were never gated by it (#550).
- A real setting with no documentation row. Credentials, CORS origins, and ACL
  rules were all reachable and undocumented, which is how you end up with a
  deployment configured from source-reading.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from strata.config import StrataConfig

_REPO = Path(__file__).resolve().parent.parent
_DOC = _REPO / "docs" / "reference" / "configuration.md"
_SRC = _REPO / "src"

# Vars read straight from ``os.environ`` rather than declared on StrataConfig
# (logging and tracing initialize before config exists; the worker vars are
# read by ``strata-worker``, a separate process with no StrataConfig at all).
_ENV_LOOKUP = re.compile(r"""os\.environ(?:\.get)?[(\[]\s*["'](STRATA_[A-Z0-9_]+)["']""")
_DOCUMENTED = re.compile(r"`(STRATA_[A-Z0-9_]+)`")


def _documented() -> set[str]:
    return set(_DOCUMENTED.findall(_DOC.read_text()))


def _from_config() -> set[str]:
    return {f"STRATA_{name.upper()}" for name in StrataConfig.model_fields}


def _from_environ_lookups() -> set[str]:
    found: set[str] = set()
    for path in _SRC.rglob("*.py"):
        found |= set(_ENV_LOOKUP.findall(path.read_text()))
    return found


@pytest.fixture(scope="module")
def real() -> set[str]:
    return _from_config() | _from_environ_lookups()


def test_every_documented_variable_exists(real):
    phantom = sorted(_documented() - real)
    assert not phantom, (
        f"Documented but not read anywhere: {phantom}. StrataConfig ignores "
        "unknown STRATA_* vars, so setting one of these does nothing at all "
        "and says nothing about it."
    )


def test_every_config_field_is_documented():
    undocumented = sorted(_from_config() - _documented())
    assert not undocumented, (
        f"Real settings with no row in {_DOC.relative_to(_REPO)}: {undocumented}"
    )


def test_the_scan_finds_the_variables_it_claims_to():
    # Guards the regexes themselves: a pattern that silently matched nothing
    # would make both assertions above pass forever.
    assert "STRATA_HOST" in _from_config()
    assert "STRATA_LOG_LEVEL" in _from_environ_lookups()
    assert "STRATA_HOST" in _documented()
