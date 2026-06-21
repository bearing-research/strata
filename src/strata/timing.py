"""Shared timing helpers.

A single home for measuring elapsed wall-clock time in milliseconds, so the
``(time.perf_counter() - start) * 1000`` idiom isn't re-spelled at every call
site. Specialized timers that record into a tracker (``slow_ops``) or collect
named phases (``notebook.timing``) build on top of this.
"""

from __future__ import annotations

import time


def elapsed_ms(start: float) -> float:
    """Return milliseconds elapsed since a ``time.perf_counter`` mark.

    Parameters
    ----------
    start : float
        A value captured earlier from :func:`time.perf_counter`.

    Returns
    -------
    float
        Elapsed time in milliseconds.
    """
    return (time.perf_counter() - start) * 1000


class Timer:
    """Context manager that measures wall-clock duration in milliseconds.

    On exit, ``elapsed_ms`` holds the time spent in the ``with`` block.

    Examples
    --------
    >>> with Timer() as t:
    ...     do_work()
    >>> t.elapsed_ms
    """

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
