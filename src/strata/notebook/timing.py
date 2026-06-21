"""Notebook request timing helpers."""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from strata.timing import elapsed_ms


def _sanitize_server_timing_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", name.strip())
    return sanitized or "phase"


@dataclass
class NotebookTimingRecorder:
    """Collect lightweight phase timings for notebook request handling."""

    _started_at: float = field(default_factory=time.perf_counter)
    _phases_ms: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self._phases_ms[name] = elapsed_ms(started)

    def record_duration_ms(self, name: str, duration_ms: float | int | None) -> None:
        if duration_ms is None:
            return
        self._phases_ms[name] = float(duration_ms)

    def as_dict(self, *, include_total: bool = True) -> dict[str, float]:
        timings = dict(self._phases_ms)
        if include_total:
            timings["total"] = elapsed_ms(self._started_at)
        return timings

    def server_timing_header(self) -> str:
        return ", ".join(
            f"{_sanitize_server_timing_name(name)};dur={duration_ms:.1f}"
            for name, duration_ms in self.as_dict().items()
        )
