"""Wall-clock timing helpers in milliseconds."""

from __future__ import annotations

import time


def elapsed_ms(start: float) -> float:
    """Return milliseconds elapsed since a ``time.perf_counter()`` mark."""
    return (time.perf_counter() - start) * 1000


class Timer:
    """Context manager; on exit ``elapsed_ms`` holds the ``with`` block's duration."""

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> Timer:
        """Start the timer and return self."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        """Stop the timer, recording the elapsed milliseconds."""
        self.elapsed_ms = elapsed_ms(self.start_time)
