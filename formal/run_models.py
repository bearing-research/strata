"""Run every TLA+ configuration and check it ends the way MODELS says.

A configuration either passes (no invariant violated, exhaustively) or shows a
named violation: a model of the code before a fix. Both outcomes are pinned, so
a model that stops finding its bug fails here as surely as a patched one that
starts violating.

    uv run python formal/run_models.py            # all
    uv run python formal/run_models.py Admission  # one spec's configurations

TLC comes from the pinned tla2tools release, cached under ~/.cache/strata-formal
(or $TLA2TOOLS to point at a jar). Needs Java 11+.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

TLA_VERSION = "v1.7.4"  # TLC 2.19
TLA_SHA256 = "936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88"
TLA_URL = f"https://github.com/tlaplus/tlaplus/releases/download/{TLA_VERSION}/tla2tools.jar"

TLA_DIR = Path(__file__).parent / "tla"

# (spec, config): None passes; a name is the invariant the configuration violates.
MODELS: dict[tuple[str, str], str | None] = {
    ("Admission", "Admission_Py312"): "NoLostWakeup",
    ("Admission", "Admission_Eviction"): "DrainSeesAll",
    ("Admission", "Admission_Patched"): None,
    ("Staleness", "Staleness_EditDuringRun"): "RunningShown",
    ("Staleness", "Staleness_Patched"): None,
    ("LocalRun", "LocalRun_Lapse"): "NoNeedlessFence",
    ("LocalRun", "LocalRun_Race"): "NoNeedlessFence",
    ("LocalRun", "LocalRun_Patched"): None,
}


def tla2tools() -> Path:
    if env := os.environ.get("TLA2TOOLS"):
        return Path(env)
    jar = Path.home() / ".cache" / "strata-formal" / f"tla2tools-{TLA_VERSION}.jar"
    if not jar.exists():
        jar.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(TLA_URL) as response:
            data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != TLA_SHA256:
            raise SystemExit(f"tla2tools.jar sha256 {digest}, expected {TLA_SHA256}")
        jar.write_bytes(data)
    return jar


def check(jar: Path, spec: str, config: str, expected: str | None) -> str | None:
    """Run one configuration; return a complaint, or None when it ends as expected."""
    # One worker when a violation is expected: TLC reports the first it finds,
    # and with several workers which one comes first varies.
    workers = "auto" if expected is None else "1"
    with tempfile.TemporaryDirectory() as meta:
        result = subprocess.run(
            [
                "java",
                "-XX:+UseParallelGC",
                "-cp",
                str(jar),
                "tlc2.TLC",
                "-deadlock",
                "-workers",
                workers,
                "-metadir",
                meta,
                "-config",
                f"{config}.cfg",
                f"{spec}.tla",
            ],
            cwd=TLA_DIR,
            capture_output=True,
            text=True,
        )
    out = result.stdout + result.stderr
    states = re.search(r"([\d,]+) distinct states found", out)
    print(f"  {config}: {states.group(1) if states else '?'} distinct states")
    violated = re.search(r"Invariant (\w+) is violated", out)
    if expected is None:
        if "Model checking completed. No error has been found." in out:
            return None
        return f"{config}: expected to pass\n{out[-3000:]}"
    if violated and violated.group(1) == expected:
        return None
    return f"{config}: expected {expected} violated\n{out[-3000:]}"


def main(argv: list[str]) -> int:
    if shutil.which("java") is None:
        print("java not found; TLC needs Java 11+", file=sys.stderr)
        return 2
    jar = tla2tools()
    selected = {k: v for k, v in MODELS.items() if not argv or k[0] in argv}
    unlisted = {p.stem for p in TLA_DIR.glob("*.cfg")} - {c for _, c in MODELS}
    failures = [f"{name}.cfg is not listed in MODELS" for name in sorted(unlisted)]
    for (spec, config), expected in selected.items():
        print(f"{spec}.tla {'passes' if expected is None else 'violates ' + expected}")
        if complaint := check(jar, spec, config, expected):
            failures.append(complaint)
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
