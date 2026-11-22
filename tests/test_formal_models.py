"""The TLA+ models under formal/ name the code they mirror, and that code exists.

A model checks the code only while it matches it. Each spec lists what it
models in a ``Mirrors:`` block (``strata/<path>.py::Qualname``); when one of
those moves or is renamed, this fails, so the change that moved it also looks
at the model.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPECS = sorted((ROOT / "formal" / "tla").glob("*.tla"))
MIRROR = re.compile(r"(strata/[\w/]+\.py)::([\w.]+)")


def _mirrors(spec: Path) -> list[tuple[str, str]]:
    return MIRROR.findall(spec.read_text())


def _defines(module: ast.Module, qualname: str) -> bool:
    scope: list[ast.stmt] = module.body
    for part in qualname.split("."):
        found = next(
            (
                node
                for node in scope
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
                and node.name == part
            ),
            None,
        )
        if found is None:
            return False
        scope = found.body
    return True


def test_there_are_specs():
    assert SPECS


@pytest.mark.parametrize("spec", SPECS, ids=lambda p: p.stem)
def test_a_spec_names_what_it_mirrors(spec: Path):
    assert _mirrors(spec), f"{spec.name} has no Mirrors: block"


@pytest.mark.parametrize("spec", SPECS, ids=lambda p: p.stem)
def test_what_a_spec_mirrors_exists(spec: Path):
    missing = []
    for path, qualname in _mirrors(spec):
        source = ROOT / "src" / path
        if not source.exists():
            missing.append(f"{path} (no such file)")
        elif not _defines(ast.parse(source.read_text()), qualname):
            missing.append(f"{path}::{qualname}")
    assert not missing, f"{spec.name} mirrors code that is gone: {missing}"


def test_a_moved_function_is_noticed():
    module = ast.parse("class A:\n    async def f(self): ...\ndef g(): ...\n")
    assert _defines(module, "A.f")
    assert _defines(module, "g")
    assert not _defines(module, "A.g")
    assert not _defines(module, "f")
