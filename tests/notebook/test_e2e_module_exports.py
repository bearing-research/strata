"""E2E tests: cross-cell source-backed module exports."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.notebook.e2e_fixtures import (
    NotebookBuilder,
    create_test_app,
    execute_cell_and_wait,
    open_notebook_session,
    ws_connect,
)


@pytest.fixture
def setup():
    """Create app, client, and temp directory."""
    app = create_test_app()
    client = TestClient(app)
    with tempfile.TemporaryDirectory() as tmpdir:
        yield client, Path(tmpdir)


def _put_notebook_workers(client: TestClient, session_id: str, workers: list[dict]) -> dict:
    response = client.put(
        f"/v1/notebooks/{session_id}/workers",
        json={"workers": workers},
    )
    assert response.status_code == 200
    return response.json()


def _put_notebook_worker(client: TestClient, session_id: str, worker: str | None) -> dict:
    response = client.put(
        f"/v1/notebooks/{session_id}/worker",
        json={"worker": worker},
    )
    assert response.status_code == 200
    return response.json()


class TestLocalModuleExports:
    """Local execution should support source-backed exports across cells."""

    def test_cross_cell_function_export(self, setup):
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell(
                "c1",
                "import math\n\ndef area(r):\n    return math.pi * r * r",
            )
            .add_cell("c2", "result = round(area(2), 5)", after="c1")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                result1 = execute_cell_and_wait(ws, "c1")
                assert result1["type"] == "cell_output"
                assert result1["payload"]["outputs"]["area"]["content_type"] == "module/cell"

                result2 = execute_cell_and_wait(ws, "c2")
                assert result2["type"] == "cell_output"
                assert result2["payload"]["outputs"]["result"]["preview"] == 12.56637

    def test_cross_cell_runtime_value_injection(self, setup):
        """A def closing over an UPSTREAM runtime value exports via injection,
        and the downstream consumer gets that value hydrated. This pattern was
        previously rejected ("cannot be shared across cells yet")."""
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell("c1", "base = 10 + len('abc')")  # runtime value = 13
            .add_cell("c2", "def addbase(y):\n    return base + y", after="c1")
            .add_cell("c3", "result = addbase(5)", after="c2")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")
                result2 = execute_cell_and_wait(ws, "c2")
                assert result2["type"] == "cell_output"
                # The def exports as a synthetic module despite closing over `base`.
                assert result2["payload"]["outputs"]["addbase"]["content_type"] == "module/cell"

                result3 = execute_cell_and_wait(ws, "c3")
                assert result3["type"] == "cell_output"
                assert result3["payload"]["outputs"]["result"]["preview"] == 18

    def test_same_cell_runtime_value_injection(self, setup):
        """A def closing over a SAME-CELL runtime value is shareable: the value
        is stored and hydrated into the synthetic module (Phase 2)."""
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell("c1", "data = list(range(5))\n\ndef total(e):\n    return sum(data) + e")
            .add_cell("c2", "result = total(100)", after="c1")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                result1 = execute_cell_and_wait(ws, "c1")
                assert result1["type"] == "cell_output"
                assert result1["payload"]["outputs"]["total"]["content_type"] == "module/cell"

                result2 = execute_cell_and_wait(ws, "c2")
                assert result2["type"] == "cell_output"
                assert result2["payload"]["outputs"]["result"]["preview"] == 110

    def test_cross_cell_class_export(self, setup):
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell(
                "c1",
                "class Box:\n    def __init__(self, value):\n        self.value = value\n",
            )
            .add_cell("c2", "result = Box(7).value", after="c1")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                result1 = execute_cell_and_wait(ws, "c1")
                assert result1["type"] == "cell_output"
                assert result1["payload"]["outputs"]["Box"]["content_type"] == "module/cell"

                result2 = execute_cell_and_wait(ws, "c2")
                assert result2["type"] == "cell_output"
                assert result2["payload"]["outputs"]["result"]["preview"] == 7

    def test_literal_constant_coexists_with_def_in_module_cell(self, setup):
        """A literal constant alongside a def should export as part of
        the same module — no cell split required. Both names become
        available downstream."""
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell(
                "c1",
                "STEP_SIZE = 0.5\n\ndef scaled(x):\n    return x * STEP_SIZE\n",
            )
            .add_cell("c2", "result = scaled(10) + STEP_SIZE", after="c1")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                result1 = execute_cell_and_wait(ws, "c1")
                assert result1["type"] == "cell_output"
                # Both the def and the literal-constant ride the same
                # synthetic module.
                assert result1["payload"]["outputs"]["scaled"]["content_type"] == "module/cell"
                assert result1["payload"]["outputs"]["STEP_SIZE"]["content_type"] == "module/cell"

                result2 = execute_cell_and_wait(ws, "c2")
                assert result2["type"] == "cell_output"
                # Downstream resolves STEP_SIZE to 0.5 and scaled(10) to 5.0, sum = 5.5
                assert result2["payload"]["outputs"]["result"]["preview"] == 5.5

    def test_pure_constant_cell_uses_normal_artifact_path(self, setup):
        """A cell that defines only a literal constant (no defs/classes)
        should serialize through the normal artifact path — it's plain
        data, not code, and shouldn't be wrapped in a synthetic module."""
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell("c1", "THRESHOLD = 42\n")
            .add_cell("c2", "result = THRESHOLD * 2\n", after="c1")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                result1 = execute_cell_and_wait(ws, "c1")
                assert result1["type"] == "cell_output"
                # json/object, not module/cell — constant flows as data.
                assert result1["payload"]["outputs"]["THRESHOLD"]["content_type"] != "module/cell"
                assert result1["payload"]["outputs"]["THRESHOLD"]["preview"] == 42

                result2 = execute_cell_and_wait(ws, "c2")
                assert result2["type"] == "cell_output"
                assert result2["payload"]["outputs"]["result"]["preview"] == 84

    def test_cross_cell_def_export_with_runtime_state_alongside(self, setup):
        """The producing cell mixes a runtime statement with a self-
        contained def. Slicing should let the def export cleanly while
        the runtime variable flows through the regular artifact path.
        """
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell(
                "c1",
                """
df_size = len([1, 2, 3, 4, 5])
THRESHOLD = 3

def is_big(n):
    return n > THRESHOLD
""".strip(),
            )
            # Downstream consumes both the runtime variable (df_size,
            # via the regular artifact path) and the def (via module
            # export). THRESHOLD is referenced in c2 to verify the
            # literal-const-alongside-def path still works under
            # slicing.
            .add_cell(
                "c2",
                "result = [df_size, is_big(df_size), THRESHOLD]",
                after="c1",
            )
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                result1 = execute_cell_and_wait(ws, "c1")
                assert result1["type"] == "cell_output"
                # is_big and THRESHOLD ride the synthetic module
                # because c2 consumes them and a def is also kept.
                assert result1["payload"]["outputs"]["is_big"]["content_type"] == "module/cell"
                assert result1["payload"]["outputs"]["THRESHOLD"]["content_type"] == "module/cell"
                # df_size flows through the regular artifact path.
                assert result1["payload"]["outputs"]["df_size"]["content_type"] != "module/cell"
                assert result1["payload"]["outputs"]["df_size"]["preview"] == 5

                result2 = execute_cell_and_wait(ws, "c2")
                assert result2["type"] == "cell_output"
                # is_big(5) > THRESHOLD(3) → True; result is [5, True, 3]
                assert result2["payload"]["outputs"]["result"]["preview"] == [5, True, 3]

    def test_cross_cell_exported_class_instance(self, setup):
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell(
                "c1",
                "class Person:\n"
                "    name = 'John'\n"
                "    age = 20\n"
                "\n"
                "    def __str__(self):\n"
                "        return f'{self.name}:{self.age}'\n",
            )
            .add_cell("c2", "p = Person()", after="c1")
            .add_cell("c3", "rendered = str(p)", after="c2")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                result1 = execute_cell_and_wait(ws, "c1")
                assert result1["type"] == "cell_output"
                assert result1["payload"]["outputs"]["Person"]["content_type"] == "module/cell"

                result2 = execute_cell_and_wait(ws, "c2")
                assert result2["type"] == "cell_output"
                assert result2["payload"]["outputs"]["p"]["content_type"] == "module/cell-instance"

                result3 = execute_cell_and_wait(ws, "c3")
                assert result3["type"] == "cell_output"
                assert result3["payload"]["outputs"]["rendered"]["preview"] == "John:20"


class TestDirectHttpModuleExports:
    """Direct HTTP workers should preserve source-backed exports too."""

    def test_function_export_over_direct_http_worker(self, setup, notebook_executor_server):
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell(
                "c1",
                "import math\n\ndef area(r):\n    return math.pi * r * r",
            )
            .add_cell("c2", "result = round(area(2), 5)", after="c1")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            catalog = _put_notebook_workers(
                client,
                sid,
                [
                    {
                        "name": "gpu-http",
                        "backend": "executor",
                        "runtime_id": "gpu-http-a100",
                        "config": {"url": notebook_executor_server["execute_url"]},
                    }
                ],
            )
            assert any(worker["name"] == "gpu-http" for worker in catalog["workers"])
            _put_notebook_worker(client, sid, "gpu-http")

            with ws_connect(client, sid) as ws:
                result1 = execute_cell_and_wait(ws, "c1")
                assert result1["type"] == "cell_output"
                assert result1["payload"]["execution_method"] == "executor"
                assert result1["payload"]["remote_worker"] == "gpu-http"
                assert result1["payload"]["remote_transport"] == "direct"
                assert result1["payload"]["outputs"]["area"]["content_type"] == "module/cell"

                result2 = execute_cell_and_wait(ws, "c2")
                assert result2["type"] == "cell_output"
                assert result2["payload"]["execution_method"] == "executor"
                assert result2["payload"]["remote_worker"] == "gpu-http"
                assert result2["payload"]["remote_transport"] == "direct"
                assert result2["payload"]["outputs"]["result"]["preview"] == 12.56637

    def test_class_instance_export_over_direct_http_worker(
        self,
        setup,
        notebook_executor_server,
    ):
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell(
                "c1",
                "class Person:\n"
                "    name = 'John'\n"
                "    age = 20\n"
                "\n"
                "    def __str__(self):\n"
                "        return f'{self.name}:{self.age}'\n",
            )
            .add_cell("c2", "p = Person()", after="c1")
            .add_cell("c3", "rendered = str(p)", after="c2")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            catalog = _put_notebook_workers(
                client,
                sid,
                [
                    {
                        "name": "gpu-http",
                        "backend": "executor",
                        "runtime_id": "gpu-http-a100",
                        "config": {"url": notebook_executor_server["execute_url"]},
                    }
                ],
            )
            assert any(worker["name"] == "gpu-http" for worker in catalog["workers"])
            _put_notebook_worker(client, sid, "gpu-http")

            with ws_connect(client, sid) as ws:
                result1 = execute_cell_and_wait(ws, "c1")
                result2 = execute_cell_and_wait(ws, "c2")
                result3 = execute_cell_and_wait(ws, "c3")

                assert result1["type"] == "cell_output"
                assert result1["payload"]["remote_transport"] == "direct"
                assert result1["payload"]["outputs"]["Person"]["content_type"] == "module/cell"

                assert result2["type"] == "cell_output"
                assert result2["payload"]["remote_transport"] == "direct"
                assert result2["payload"]["outputs"]["p"]["content_type"] == "module/cell-instance"

                assert result3["type"] == "cell_output"
                assert result3["payload"]["remote_transport"] == "direct"
                assert result3["payload"]["outputs"]["rendered"]["preview"] == "John:20"
