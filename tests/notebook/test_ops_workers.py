"""Tests for the worker-registration verbs (LocalNotebookOps + `strata worker`).

Local backend, offline — writes notebook-scoped ``[[workers]]`` to notebook.toml
and reads them back. These are the SSH-free P0 primitives: a programmatic way to
register a worker and set the default, which the CLI and the MCP tools share.
"""

from __future__ import annotations

import json

import httpx
import pytest

from strata.cli import main
from strata.notebook.ops import (
    LocalNotebookOps,
    NotebookOpsError,
    RemoteNotebookOps,
    WorkerListView,
)
from strata.notebook.parser import parse_notebook
from tests.notebook.test_cli import _build_notebook

URL = "http://127.0.0.1:9000/v1/execute"


@pytest.fixture
def nb(tmp_path):
    return _build_notebook(tmp_path, cells=[("a", "x = 1", None)])


def test_list_workers_has_builtin_local_default(nb):
    view = LocalNotebookOps(nb).list_workers()
    assert isinstance(view, WorkerListView)
    assert view.default is None  # implicit local
    assert view.editable is True
    assert [w.name for w in view.workers] == ["local"]
    local = view.workers[0]
    assert local.backend == "local"
    assert local.transport == "local"
    assert local.is_default is True  # None default ⇒ local is the effective default


def test_add_worker_appends_and_persists(nb):
    view = LocalNotebookOps(nb).add_worker(
        "gpu", url=URL, token_env="STRATA_WORKER_TOKEN_GPU", set_default=True
    )
    assert view.default == "gpu"
    names = [w.name for w in view.workers]
    assert names == ["local", "gpu"]
    gpu = next(w for w in view.workers if w.name == "gpu")
    assert gpu.backend == "executor"
    assert gpu.transport == "direct"
    assert gpu.url == URL
    assert gpu.token_env == "STRATA_WORKER_TOKEN_GPU"
    assert gpu.is_default is True
    # Local is no longer the default once an explicit default is set.
    assert next(w for w in view.workers if w.name == "local").is_default is False

    # Persisted to notebook.toml — a fresh parse sees it.
    state = parse_notebook(nb)
    assert state.worker == "gpu"
    assert [w.name for w in state.workers] == ["gpu"]
    assert state.workers[0].config.url == URL


def test_add_worker_replaces_by_name(nb):
    ops = LocalNotebookOps(nb)
    ops.add_worker("gpu", url=URL)
    view = ops.add_worker("gpu", url="http://127.0.0.1:9001/v1/execute")
    gpus = [w for w in view.workers if w.name == "gpu"]
    assert len(gpus) == 1
    assert gpus[0].url == "http://127.0.0.1:9001/v1/execute"


def test_add_executor_worker_requires_url(nb):
    with pytest.raises(NotebookOpsError, match="requires a url"):
        LocalNotebookOps(nb).add_worker("gpu", url=None)


def test_add_worker_rejects_bad_name(nb):
    with pytest.raises(NotebookOpsError, match="invalid worker"):
        LocalNotebookOps(nb).add_worker("bad name!", url=URL)


def test_remove_worker_missing_and_builtin(nb):
    ops = LocalNotebookOps(nb)
    with pytest.raises(NotebookOpsError, match="no worker named"):
        ops.remove_worker("ghost")
    with pytest.raises(NotebookOpsError, match="built-in 'local'"):
        ops.remove_worker("local")


def test_remove_worker_clears_default_it_named(nb):
    ops = LocalNotebookOps(nb)
    ops.add_worker("gpu", url=URL, set_default=True)
    view = ops.remove_worker("gpu")
    assert view.default is None
    assert [w.name for w in view.workers] == ["local"]
    assert parse_notebook(nb).worker is None


def test_remove_worker_keeps_unrelated_default(nb):
    ops = LocalNotebookOps(nb)
    ops.add_worker("gpu", url=URL)
    ops.add_worker("cpu", url="http://127.0.0.1:9001/v1/execute", set_default=True)
    view = ops.remove_worker("gpu")
    assert view.default == "cpu"
    assert [w.name for w in view.workers] == ["local", "cpu"]


def test_set_default_worker_unknown_and_clear(nb):
    ops = LocalNotebookOps(nb)
    with pytest.raises(NotebookOpsError, match="no worker named 'ghost'"):
        ops.set_default_worker("ghost")
    ops.add_worker("gpu", url=URL, set_default=True)
    # "local" and None both clear the persisted default back to implicit-local.
    assert ops.set_default_worker("local").default is None
    assert parse_notebook(nb).worker is None
    ops.set_default_worker("gpu")
    assert ops.set_default_worker(None).default is None


# ---------------------------------------------------------------------------
# CLI — `strata worker …`
# ---------------------------------------------------------------------------


def test_cli_worker_add_ls_default_rm(nb, capsys):
    rc = main(["worker", "add", str(nb), "gpu", "--url", URL, "--default"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["default"] == "gpu"
    assert [w["name"] for w in data["workers"]] == ["local", "gpu"]

    rc = main(["worker", "ls", str(nb)])
    assert rc == 0
    listed = json.loads(capsys.readouterr().out)
    assert next(w for w in listed["workers"] if w["name"] == "gpu")["is_default"] is True

    rc = main(["worker", "rm", str(nb), "gpu"])
    assert rc == 0
    after = json.loads(capsys.readouterr().out)
    assert [w["name"] for w in after["workers"]] == ["local"]
    assert after["default"] is None


def test_cli_worker_ls_human_marks_default(nb, capsys):
    main(["worker", "add", str(nb), "gpu", "--url", URL, "--default"])
    capsys.readouterr()
    rc = main(["worker", "ls", str(nb), "--format", "human"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "* gpu" in out
    assert "  local" in out


def test_cli_worker_error_exit_code(nb, capsys):
    rc = main(["worker", "rm", str(nb), "ghost"])
    assert rc == 1
    assert json.loads(capsys.readouterr().out)["error"]


def test_cli_worker_add_missing_url_is_usage_error(nb):
    # argparse enforces --url before ops runs → exit 2 (usage), not 1.
    with pytest.raises(SystemExit) as exc:
        main(["worker", "add", str(nb), "gpu"])
    assert exc.value.code == 2


def test_cli_worker_bad_dir(capsys, tmp_path):
    rc = main(["worker", "ls", str(tmp_path / "nope")])
    assert rc == 2
    assert "not a Strata notebook" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# RemoteNotebookOps SSH-worker verbs + `strata worker add-ssh|rm-ssh` CLI
# ---------------------------------------------------------------------------


def _ssh_ops(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return RemoteNotebookOps("http://test", "sess-1", client=client)


def test_remote_add_ssh_worker_posts_and_returns():
    seen = {}

    def handler(request):
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "worker": {
                    "name": "gpu-box",
                    "ssh_target": "user@gpu-box",
                    "executor_url": "http://127.0.0.1:6000/v1/execute",
                }
            },
        )

    data = _ssh_ops(handler).add_ssh_worker("user@gpu-box", set_default=True)
    assert seen["method"] == "POST"
    assert seen["path"] == "/v1/notebooks/sess-1/workers/ssh"
    assert seen["body"] == {"ssh_target": "user@gpu-box", "set_default": True, "install": True}
    assert data["worker"]["name"] == "gpu-box"


def test_remote_add_ssh_worker_includes_name_when_given():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"worker": {"name": "gpu"}})

    _ssh_ops(handler).add_ssh_worker("user@box", name="gpu")
    assert seen["body"]["name"] == "gpu"


def test_remote_add_ssh_worker_error_raises():
    ops = _ssh_ops(lambda req: httpx.Response(400, json={"detail": "key auth failed"}))
    with pytest.raises(NotebookOpsError, match="key auth failed"):
        ops.add_ssh_worker("user@box")


def test_remote_remove_ssh_worker():
    seen = {}

    def handler(request):
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["query"] = request.url.query.decode()
        return httpx.Response(200, json={"torn_down": True})

    data = _ssh_ops(handler).remove_ssh_worker("gpu-box", stop_remote=True)
    assert seen["method"] == "DELETE"
    assert seen["path"] == "/v1/notebooks/sess-1/workers/ssh/gpu-box"
    assert "stop_remote=true" in seen["query"]
    assert data["torn_down"] is True


def test_cli_worker_add_ssh_requires_server_and_session():
    # --server / --session are required → argparse usage error, not a run.
    with pytest.raises(SystemExit) as exc:
        main(["worker", "add-ssh", "user@box"])
    assert exc.value.code == 2


def test_cli_worker_add_ssh_happy(monkeypatch, capsys):
    calls = {}

    class FakeOps:
        def __init__(self, server, session):
            calls["init"] = (server, session)

        def add_ssh_worker(self, target, **kwargs):
            calls["add"] = (target, kwargs)
            return {"worker": {"name": "gpu-box", "ssh_target": target}}

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr("strata.notebook.ops.RemoteNotebookOps", FakeOps)
    rc = main(
        [
            "worker",
            "add-ssh",
            "user@gpu-box",
            "--server",
            "http://s",
            "--session",
            "sess",
            "--no-default",
            "--format",
            "human",
        ]
    )
    assert rc == 0
    assert calls["init"] == ("http://s", "sess")
    assert calls["add"][0] == "user@gpu-box"
    assert calls["add"][1]["set_default"] is False  # --no-default
    assert calls["closed"] is True
    assert "gpu-box" in capsys.readouterr().out
