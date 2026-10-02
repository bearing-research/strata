"""The sub-packages ship the repository's LICENSE in their wheels and sdists."""

import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("package", ["strata-client", "strata-pool"])
def test_each_sub_package_ships_the_root_license(package):
    package_dir = ROOT / "packages" / package
    with open(package_dir / "pyproject.toml", "rb") as f:
        project = tomllib.load(f)

    assert project["project"]["license-files"] == ["LICENSE"]
    assert "LICENSE" in project["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    # A copy, since a build backend cannot reach outside the project; it must not drift.
    assert (package_dir / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()
