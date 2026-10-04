"""A job's time in the pool, queueing and boot included, in its submitter's trace."""

import pytest

pytest.importorskip("opentelemetry.sdk")

from conftest import FakeBackend  # noqa: E402
from opentelemetry.propagate import inject  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode  # noqa: E402


class TickingClock:
    """Each reading a second after the last, so every recorded moment is distinct."""

    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


@pytest.fixture
def traced():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


def _one(exporter, name):
    (span,) = [s for s in exporter.get_finished_spans() if s.name == name]
    return span


def _submitted_under(tracer):
    with tracer.start_as_current_span("dispatch") as dispatch:
        carrier: dict[str, str] = {}
        inject(carrier)
    return dispatch, carrier


async def test_a_job_that_waits_for_a_boot_traces_its_queueing_and_the_boot(make_pool, traced):
    tracer, exporter = traced
    pool = make_pool(tracer=tracer, wall=TickingClock())
    dispatch, carrier = _submitted_under(tracer)

    job = await pool.submit(
        tenant_id="acme", machine_type="cpu", payload=b"x", trace_context=carrier
    )
    await pool.wait(job.id)

    queued, boot, execution = (
        _one(exporter, name) for name in ("pool.queue", "pool.boot", "pool.execute")
    )
    for span in (queued, boot, execution):
        assert span.context.trace_id == dispatch.context.trace_id
        assert span.parent.span_id == dispatch.context.span_id
        assert span.attributes["job_id"] == job.id
    (worker,) = pool.store.list_workers()
    # Queueing runs from the submit to the dispatch, and the boot falls inside it.
    assert queued.start_time == int(job.submitted_at * 1e9)
    assert boot.start_time == int(worker.created_at * 1e9)
    assert queued.start_time < boot.start_time < boot.end_time <= queued.end_time
    assert boot.attributes["worker_id"] == worker.id
    assert boot.status.status_code is StatusCode.UNSET


async def test_a_machine_that_fails_to_start_is_an_error_in_the_waiting_jobs_trace(
    make_pool, traced
):
    tracer, exporter = traced
    pool = make_pool(backend=FakeBackend(fail_start=True), tracer=tracer)
    dispatch, carrier = _submitted_under(tracer)

    job = await pool.submit(
        tenant_id="acme", machine_type="cpu", payload=b"x", trace_context=carrier
    )

    boot = _one(exporter, "pool.boot")
    assert boot.parent.span_id == dispatch.context.span_id
    assert boot.attributes["job_id"] == job.id
    assert boot.status.status_code is StatusCode.ERROR
