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
from strata.notebook.remote_executor import _WORKER_SECRETS
from strata.notebook.serializer import OBJECT_CODEC_ENV_VAR
from strata.notebook.worker_env import ENV_ROOT_VAR, REGISTRY_VAR

_REPO = Path(__file__).resolve().parent.parent
_DOC = _REPO / "docs" / "reference" / "configuration.md"
_SRC = _REPO / "src"
_CLIENT_SRC = _REPO / "packages" / "strata-client" / "src"

# Read straight from ``os.environ``, not declared on StrataConfig: logging and
# tracing initialize before config exists, and ``strata-worker`` has no StrataConfig.
_ENV_LOOKUP = re.compile(
    r"""(?:os\.environ(?:\.get)?[(\[]|os\.getenv\(|_positive_int_env\()\s*["'](STRATA_[A-Z0-9_]+)["']"""
)
_DOCUMENTED = re.compile(r"`(STRATA_[A-Z0-9_]+)`")

# Internal plumbing the parent hands its harness child for batch execution;
# not operator settings.
_NOT_OPERATOR_FACING = {
    "STRATA_BATCH_FRAME_FD",
    "STRATA_BATCH_RESP_FD",
    "STRATA_BATCH_OUTPUT_DIR",
}


def _read(path: Path) -> str:
    # Explicit encoding: the default is locale-dependent and two TUI modules carry
    # non-cp1252 bytes, which breaks Windows.
    return path.read_text(encoding="utf-8")


def _documented() -> set[str]:
    return set(_DOCUMENTED.findall(_read(_DOC)))


def _from_config() -> set[str]:
    return {f"STRATA_{name.upper()}" for name in StrataConfig.model_fields}


def _from_environ_lookups() -> set[str]:
    found: set[str] = set()
    for root in (_SRC, _CLIENT_SRC):
        for path in root.rglob("*.py"):
            found |= set(_ENV_LOOKUP.findall(_read(path)))
    # A worker reads its secrets from the environment once at startup, so the only
    # ``os.environ`` call naming them takes the name as a variable.
    found |= set(_WORKER_SECRETS)
    # These settings are read through named constants.
    found |= {ENV_ROOT_VAR, REGISTRY_VAR, OBJECT_CODEC_ENV_VAR}
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


def test_every_real_variable_is_documented(real):
    undocumented = sorted(real - _documented() - _NOT_OPERATOR_FACING)
    assert not undocumented, (
        f"Real settings with no row in {_DOC.relative_to(_REPO)}: {undocumented}"
    )


def test_the_scan_finds_the_variables_it_claims_to():
    # Guards the regexes: a pattern that matched nothing would make both
    # assertions above pass forever.
    assert "STRATA_HOST" in _from_config()
    assert "STRATA_LOG_LEVEL" in _from_environ_lookups()
    assert "STRATA_SERVER_URL" in _from_environ_lookups()
    assert "STRATA_HOST" in _documented()


def _service_mode_doc() -> str:
    """The service-mode page, with newlines normalized to LF."""
    path = Path(__file__).parent.parent / "docs/deployment/service-mode.md"
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_the_documented_acl_example_actually_loads():
    """The ACL block in service-mode.md must be a config, not a plausible one.

    It was neither: it used ``resource`` and ``scope``, which are not fields,
    while the real rule is ``principal`` / ``tenant`` / ``tables`` — so the
    documented example failed validation at startup. And it announced the
    default for an unmatched request as ``deny`` when the code's default is
    ``allow``, which is wrong in the direction an operator pays for: they read
    it, believe unmatched tables are refused, and ship a store that serves
    them.

    Nothing checked it, which is why it stayed wrong. This checks it.
    """
    import tomllib

    from strata.config import AclConfig

    # Normalize newlines: on a CRLF checkout a regex anchored on "\n" matches
    # nothing and the test passes by finding no block.
    doc = _service_mode_doc()
    block = re.search(r"```toml\n(\[tool\.strata\.acl_config\].*?)```", doc, re.DOTALL)
    assert block is not None, "the ACL example block is gone or no longer TOML"

    # The doc shows the pyproject-nested form; load it as the config sees it.
    parsed = tomllib.loads(block.group(1))["tool"]["strata"]["acl_config"]
    config = AclConfig(**parsed)

    assert config.default in ("allow", "deny")
    assert config.deny_rules and config.allow_rules, "the example should show both directions"
    for rule in [*config.deny_rules, *config.allow_rules]:
        assert rule.tables, "a rule with no table patterns can never match"


def test_the_documented_acl_default_matches_the_code():
    """Whatever the example sets, the prose about the *default* must be true."""
    from strata.config import AclConfig

    doc = _service_mode_doc()

    assert f'Defaults to "{AclConfig().default}"' in doc, (
        "service-mode.md states a default for unmatched ACL requests that the code disagrees with"
    )
