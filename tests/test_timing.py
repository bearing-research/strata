"""Tests for the shared timing helpers."""

from __future__ import annotations

import time

from strata.timing import Timer, elapsed_ms


def test_elapsed_ms_returns_milliseconds():
    start = time.perf_counter()
    # Tiny but non-zero; assert correctness only, no timing thresholds.
    time.sleep(0.005)
    ms = elapsed_ms(start)
    assert isinstance(ms, float)
    assert ms > 0


def test_elapsed_ms_zero_for_now():
    assert elapsed_ms(time.perf_counter()) >= 0


def test_timer_records_elapsed_on_exit():
    with Timer() as t:
        time.sleep(0.005)
    assert isinstance(t.elapsed_ms, float)
    assert t.elapsed_ms > 0


def test_timer_elapsed_is_zero_before_exit():
    t = Timer()
    assert t.elapsed_ms == 0.0
    with t:
        pass
    assert t.elapsed_ms >= 0.0
