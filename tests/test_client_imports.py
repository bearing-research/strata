"""Guard the slim-client import boundary.

``import strata`` / ``import strata.client`` is the surface the future
standalone ``strata-client`` distribution ships. It must not pull in the
server's heavy dependency stack — duckdb, fastapi/uvicorn, pyiceberg, or the
pydantic config layer. These run in a *fresh* interpreter subprocess so the
test's own (server-laden) session doesn't pre-populate ``sys.modules``.

See docs/internal/design-strata-client.md.
"""

from __future__ import annotations

import subprocess
import sys

_HEAVY_SERVER_DEPS = ("duckdb", "fastapi", "uvicorn", "pyiceberg", "pydantic")


def _modules_after_import(import_stmt: str) -> set[str]:
    """Return the heavy server modules present after ``import_stmt`` in a clean interpreter."""
    code = (
        f"{import_stmt}\n"
        "import sys\n"
        f"heavy = {_HEAVY_SERVER_DEPS!r}\n"
        "present = sorted(m for m in heavy if m in sys.modules)\n"
        "print(','.join(present))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"import failed:\n{result.stderr}"
    out = result.stdout.strip()
    return set(out.split(",")) if out else set()


def test_import_strata_does_not_pull_server_deps() -> None:
    pulled = _modules_after_import("import strata")
    assert not pulled, f"`import strata` pulled heavy server deps: {sorted(pulled)}"


def test_import_strata_client_does_not_pull_server_deps() -> None:
    pulled = _modules_after_import("import strata.client")
    assert not pulled, f"`import strata.client` pulled heavy server deps: {sorted(pulled)}"


def test_client_constructs_without_loading_full_config() -> None:
    """A default StrataClient resolves its URL via the slim resolver, no StrataConfig."""
    code = (
        "import strata\n"
        "c = strata.StrataClient()\n"
        "assert c.config is None, c.config\n"
        "assert c.base_url == 'http://127.0.0.1:8765', c.base_url\n"
        "import sys\n"
        "assert 'pydantic' not in sys.modules, 'constructing the client pulled pydantic'\n"
        "c.close()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"client construction failed:\n{result.stderr}"
