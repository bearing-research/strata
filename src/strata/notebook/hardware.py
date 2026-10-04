"""What hardware this process runs on, as the machine itself reports it.

The machine type (a class like "a100-80gb") is part of a cell's identity; the
hardware is recorded beside it, not hashed, so machines of a class share a cache.
A field that could not be read is left out: absent means unknown, never "none".
Standard library only.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from typing import Any

_NVIDIA_SMI_TIMEOUT_SECONDS = 10.0
_CUDA_VERSION = re.compile(r"CUDA Version:\s*([0-9.]+)")


@functools.cache
def hardware_report() -> dict[str, Any]:
    """This machine's hardware, probed once per process."""
    return probe_hardware()


def probe_hardware() -> dict[str, Any]:
    report: dict[str, Any] = {}
    cpus = _cpus()
    if cpus:
        report["cpus"] = cpus
    memory_mb = _memory_mb()
    if memory_mb:
        report["memory_mb"] = memory_mb
    accelerators, cuda = _nvidia()
    if accelerators:
        report["accelerators"] = accelerators
    if cuda:
        report["cuda"] = cuda
    return report


def _cpus() -> int | None:
    # Affinity reflects a container's CPU limit; the host count does not.
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


def _memory_mb() -> int | None:
    # sysconf is POSIX only; Windows reports no memory rather than failing the probe.
    if not hasattr(os, "sysconf"):
        return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024 * 1024)
    except (ValueError, OSError):
        return None


def _nvidia() -> tuple[list[dict[str, Any]], str | None]:
    """GPUs, and the CUDA version the driver supports, from ``nvidia-smi``."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return [], None
    listing = _run(
        [
            smi,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    accelerators = []
    for line in (listing or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3 or not parts[0]:
            continue
        name, memory, driver = parts
        accelerator: dict[str, Any] = {"name": name}
        if memory.isdigit():
            accelerator["memory_mb"] = int(memory)
        if driver:
            accelerator["driver"] = driver
        accelerators.append(accelerator)
    summary = _run([smi]) if accelerators else None
    match = _CUDA_VERSION.search(summary or "")
    return accelerators, match.group(1) if match else None


def _run(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout if completed.returncode == 0 else None
