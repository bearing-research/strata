"""Print a wheel's requirements (no extras), one per line, leaving out the named ones.

    python scripts/wheel_requirements.py dist/strata_notebook-*.whl --without uv

Standard library only: it runs in an environment the wheel is not installed in yet.
"""

from __future__ import annotations

import argparse
import re
import zipfile
from email.parser import BytesParser


def _name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9._-]+", requirement)
    assert match, requirement
    return re.sub(r"[-_.]+", "-", match.group()).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel")
    parser.add_argument("--without", action="append", default=[])
    args = parser.parse_args()
    left_out = {_name(n) for n in args.without}
    with zipfile.ZipFile(args.wheel) as wheel:
        metadata = next(n for n in wheel.namelist() if n.endswith(".dist-info/METADATA"))
        message = BytesParser().parsebytes(wheel.read(metadata))
    for requirement in message.get_all("Requires-Dist") or []:
        if "extra ==" not in requirement and _name(requirement) not in left_out:
            print(requirement)


if __name__ == "__main__":
    main()
