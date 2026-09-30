"""Naming the hosts a worker may fetch from, instead of disabling the check.

A managed worker talks to the server that dispatched it, and that server is
usually on a private address — so the SSRF guard has to be relaxed somehow.
The only lever was STRATA_WORKER_ALLOW_LOCAL_HOSTS, which turns the address
rule off for *every* host. Item 15.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from strata.notebook.remote_executor import (
    _assert_url_safe,
    _host_is_allowlisted,
    create_notebook_executor_app,
)
from tests.notebook.test_worker_input_streaming import _manifest, _Store


@pytest.fixture(autouse=True)
def _no_ambient_bypass(monkeypatch):
    monkeypatch.delenv("STRATA_WORKER_ALLOW_LOCAL_HOSTS", raising=False)
    monkeypatch.delenv("STRATA_WORKER_ALLOWED_HOSTS", raising=False)


class TestMatching:
    @pytest.mark.parametrize(
        "allowed,host,expected",
        [
            ("build.internal", "build.internal", True),
            ("build.internal", "other.internal", False),
            (".internal", "build.internal", True),
            (".internal", "deep.build.internal", True),
            (".internal", "internal", True),
            # The one that matters: a suffix is anchored on the dot, so a
            # lookalike domain someone else registered does not match.
            (".example.com", "build.example.com", True),
            (".example.com", "evil-example.com", False),
            ("build.internal", "BUILD.INTERNAL", True),
            ("build.internal", "build.internal.", True),
            ("a.internal, b.internal", "b.internal", True),
        ],
    )
    def test_names_match_as_written(self, monkeypatch, allowed, host, expected):
        monkeypatch.setenv("STRATA_WORKER_ALLOWED_HOSTS", allowed)

        assert _host_is_allowlisted(host) is expected

    def test_nothing_is_allowlisted_by_default(self):
        assert _host_is_allowlisted("build.internal") is False


class TestGuard:
    def test_a_private_address_is_refused_without_the_allowlist(self):
        with pytest.raises(HTTPException) as excinfo:
            _assert_url_safe("http://127.0.0.1:8000/v1/builds/b1/finalize", "finalize_url")

        assert excinfo.value.status_code == 400

    def test_an_allowlisted_host_passes(self, monkeypatch):
        """Trusted because it was named, not because of where it points."""
        monkeypatch.setenv("STRATA_WORKER_ALLOWED_HOSTS", "localhost")

        _assert_url_safe("http://localhost:8000/v1/builds/b1/finalize", "finalize_url")

    def test_a_host_not_on_the_list_still_faces_the_address_rule(self, monkeypatch):
        monkeypatch.setenv("STRATA_WORKER_ALLOWED_HOSTS", "build.internal")

        with pytest.raises(HTTPException):
            _assert_url_safe("http://127.0.0.1:8000/x", "finalize_url")

    def test_the_scheme_check_still_applies_to_an_allowlisted_host(self, monkeypatch):
        """The allowlist relaxes the address rule, not the whole guard."""
        monkeypatch.setenv("STRATA_WORKER_ALLOWED_HOSTS", "localhost")

        with pytest.raises(HTTPException) as excinfo:
            _assert_url_safe("file://localhost/etc/passwd", "finalize_url")

        assert "scheme" in excinfo.value.detail


class TestBypassOrdering:
    """The wholesale bypass skipped more than it was documented to.

    It returned before the has-a-host check, so with it set — which is on
    every managed worker today, since it is the documented way to reach a
    server on a private address — a URL with no host at all was accepted.
    """

    def test_a_hostless_url_is_refused_even_with_the_bypass_set(self, monkeypatch):
        monkeypatch.setenv("STRATA_WORKER_ALLOW_LOCAL_HOSTS", "1")

        with pytest.raises(HTTPException) as excinfo:
            _assert_url_safe("http:///v1/builds/b1/finalize", "finalize_url")

        assert "missing a host" in excinfo.value.detail

    def test_the_bypass_still_relaxes_the_address_rule(self, monkeypatch):
        """What it is actually for, unchanged."""
        monkeypatch.setenv("STRATA_WORKER_ALLOW_LOCAL_HOSTS", "1")

        _assert_url_safe("http://127.0.0.1:8000/v1/builds/b1/finalize", "finalize_url")


class TestRebinding:
    def test_an_input_url_that_rebinds_to_loopback_is_not_downloaded(
        self, monkeypatch, rebinding_dns
    ):
        """The manifest is checked when it arrives and the input is fetched
        after; a name that answers the check with a public address and the
        download with 127.0.0.1 used to reach whatever listens there."""
        store = _Store(1024)
        port = store.server.server_address[1]
        manifest = _manifest(store, "x = 1")
        manifest["inputs"][0]["url"] = f"http://rebind.test:{port}/input"
        # The upload and finalize URLs stay on 127.0.0.1, named, so the guard
        # passes them and only the input's name is in question.
        monkeypatch.setenv("STRATA_WORKER_ALLOWED_HOSTS", "127.0.0.1")
        rebinding_dns.answers["rebind.test"] = [["93.184.216.34"], ["127.0.0.1"]]

        with TestClient(create_notebook_executor_app()) as worker:
            with pytest.raises(httpx.ConnectError, match="non-routable address 127.0.0.1"):
                worker.post("/v1/execute-manifest", json=manifest)

        assert store.sent == 0
        assert store.uploads == []
