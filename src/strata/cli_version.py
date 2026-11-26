"""``--version`` for Strata's commands: the installed ``strata-notebook``, read when asked."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Any


class VersionAction(argparse.Action):
    """Print ``strata-notebook <version>`` and exit; a source tree never installed says so."""

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str = argparse.SUPPRESS,
        default: Any = argparse.SUPPRESS,
        help: str = "print the installed strata-notebook version and exit",
    ) -> None:
        super().__init__(option_strings, dest=dest, default=default, nargs=0, help=help)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        try:
            installed = version("strata-notebook")
        except PackageNotFoundError:
            installed = "unknown (not installed)"
        print(f"strata-notebook {installed}")
        parser.exit()
