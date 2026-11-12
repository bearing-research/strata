"""Throwaway: where a worker's time goes under the leak gate. Not for merging."""

import gc
import os
import resource
import time

_real = {n: getattr(gc, n) for n in ("collect", "freeze", "unfreeze")}
_t = {}


def _wrap(name):
    def timed(*a, **k):
        young = "collect-young" if gc.get_freeze_count() else "collect-full"
        kind = young if name == "collect" else name
        t = time.perf_counter()
        try:
            return _real[name](*a, **k)
        finally:
            r = _t.setdefault(kind, [0.0, 0, 0.0])
            d = time.perf_counter() - t
            r[0] += d
            r[1] += 1
            r[2] = max(r[2], d)

    return timed


for _n in _real:
    setattr(gc, _n, _wrap(_n))

_phase = {"setup": 0.0, "call": 0.0, "teardown": 0.0}
_start = time.perf_counter()


def pytest_runtest_logreport(report):
    _phase[report.when] += report.duration


def pytest_sessionfinish(session):
    out = os.environ.get("GCPROBE_DIR")
    if not out:
        return
    os.makedirs(out, exist_ok=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / (1024 * 1024) if os.uname().sysname == "Darwin" else rss / 1024
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    with open(os.path.join(out, f"{worker}.txt"), "w") as f:
        count = os.environ.get("PYTEST_XDIST_WORKER_COUNT")
        wall = time.perf_counter() - _start
        f.write(f"worker={worker} of {count} cpus={os.cpu_count()} wall={wall:.0f}s ")
        f.write(f"maxrss={rss_mb:.0f}MB frozen={gc.get_freeze_count()}\n")
        for k, v in _phase.items():
            f.write(f"  phase {k}: {v:.0f}s\n")
        for k, (s, n, m) in sorted(_t.items()):
            f.write(f"  {k}: {s:.1f}s over {n}, max {m:.2f}s\n")
