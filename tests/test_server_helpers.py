import threading
from types import SimpleNamespace

import pytest

from strata.server import DRAIN_TIMEOUT_SECONDS
from tests.server_helpers import shutdown_wait_seconds, stop_strata_server


@pytest.fixture
def stuck(monkeypatch):
    """A server thread that does not stop, and a wait of a tenth of a second for it."""
    monkeypatch.setattr("strata.server.DRAIN_TIMEOUT_SECONDS", -14.9)
    release = threading.Event()
    thread = threading.Thread(target=release.wait, daemon=True)
    thread.start()
    yield thread
    release.set()
    thread.join()


def test_a_server_that_stops_is_told_to_and_waited_for():
    server = SimpleNamespace(should_exit=False)
    thread = threading.Thread(target=lambda: None)
    thread.start()

    stop_strata_server(server, thread, 1234)

    assert server.should_exit is True
    assert not thread.is_alive()


def test_a_server_still_shutting_down_is_reported(stuck):
    with pytest.raises(RuntimeError, match="port 1234 still shutting down"):
        stop_strata_server(SimpleNamespace(should_exit=False), stuck, 1234)


def test_the_report_does_not_hide_a_failure_on_its_way_out(stuck):
    # As in run_server's finally, when the body of its with block raised.
    with pytest.raises(AssertionError, match="the test's own"):
        try:
            raise AssertionError("the test's own")
        finally:
            stop_strata_server(SimpleNamespace(should_exit=False), stuck, 1234)


def test_the_wait_outlasts_the_drain_of_active_scans():
    # A shutdown still draining when the wait ends clears the next test's state.
    assert shutdown_wait_seconds() > DRAIN_TIMEOUT_SECONDS


def test_a_shutdown_that_ends_within_the_wait_is_not_reported(monkeypatch):
    # A wait of 5 s for a shutdown of a fifth of a second.
    monkeypatch.setattr("strata.server.DRAIN_TIMEOUT_SECONDS", -10)
    thread = threading.Thread(target=threading.Event().wait, args=(0.2,))
    thread.start()

    stop_strata_server(SimpleNamespace(should_exit=False), thread, 1234)

    assert not thread.is_alive()
