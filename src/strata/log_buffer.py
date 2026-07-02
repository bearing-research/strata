"""In-memory ring buffer of recent structured log entries (observability, B1).

A ``logging.Handler`` that keeps the most recent N structured entries in a
bounded deque, each tagged with a monotonic ``cursor`` — so ``GET /v1/logs`` can
page (``?since=<cursor>``) and ``GET /v1/logs/stream`` can tail (poll by cursor).
Older entries live only in the stderr stream / disk. Thread-safe, since logging
emits from executor threads too.

Leaf module: imports only ``strata.logging`` (the JSON formatter) + stdlib.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import deque
from typing import Any

from strata.logging import StructuredFormatter

# Ring-buffer retention. 10k entries is plenty for a live viewer; older entries
# remain on the stderr stream / disk (design open-question #3).
DEFAULT_CAPACITY = 10_000

# Numeric level thresholds for the ``?level=`` minimum-level filter.
_LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40, "critical": 50}

# Loggers the buffer attaches to (mirrors ``configure_logging``).
_ATTACH_LOGGERS = ("strata", "uvicorn", "uvicorn.error", "uvicorn.access")


class RingBufferLogHandler(logging.Handler):
    """Keep the most recent structured log entries in memory, cursor-tagged."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        super().__init__()
        self._formatter = StructuredFormatter()
        self._entries: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._cursor = 0

    def emit(self, record: logging.LogRecord) -> None:
        # Reuse the JSON formatter so a buffered entry is identical to the one on
        # stderr, then parse it back to a dict for structured reads.
        try:
            entry = json.loads(self._formatter.format(record))
        except Exception:  # pragma: no cover - a formatter failure must not break logging
            return
        with self._lock:
            self._cursor += 1
            entry["cursor"] = self._cursor
            self._entries.append(entry)

    def read(
        self,
        *,
        since: int = 0,
        level: str | None = None,
        notebook: str | None = None,
        regex: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        """Return entries with ``cursor > since`` matching the filters.

        Returns ``{"entries": [...], "cursor": <latest>}``. ``entries`` is the
        most recent ``limit`` matches, oldest-first. ``cursor`` is the latest
        buffered cursor (pass it back as ``since`` to page / tail). ``regex`` is a
        Python regex matched against the message; a bad pattern raises ``re.error``.
        """
        min_level = _LEVELS.get((level or "").lower(), 0)
        pattern = re.compile(regex) if regex else None

        with self._lock:
            snapshot = list(self._entries)
            latest = self._cursor

        matches: list[dict[str, Any]] = []
        for entry in snapshot:
            if entry["cursor"] <= since:
                continue
            if min_level and _LEVELS.get(str(entry.get("level", "")), 0) < min_level:
                continue
            if notebook is not None and entry.get("notebook_id") != notebook:
                continue
            if pattern is not None and not pattern.search(str(entry.get("message", ""))):
                continue
            matches.append(entry)

        return {"entries": matches[-limit:], "cursor": latest}


_ring_buffer: RingBufferLogHandler | None = None


def get_log_ring_buffer() -> RingBufferLogHandler | None:
    """Return the installed ring buffer, or ``None`` if not installed."""
    return _ring_buffer


def install_ring_buffer(capacity: int = DEFAULT_CAPACITY) -> RingBufferLogHandler:
    """Install the ring buffer on the strata + uvicorn loggers (idempotent).

    Called once at server startup — the buffer is a server-side observability
    surface, so CLI / harness processes never pay for it.
    """
    global _ring_buffer
    if _ring_buffer is None:
        _ring_buffer = RingBufferLogHandler(capacity)
        for name in _ATTACH_LOGGERS:
            logging.getLogger(name).addHandler(_ring_buffer)
    return _ring_buffer
