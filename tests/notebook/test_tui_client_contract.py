"""The TUI against a real server: the fields and frames the spectator reads are there.

The app tests replace the client and socket with fakes shaped like the responses, so a
route or frame that renamed a field would pass them and break the TUI.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
import websockets

from strata.notebook.tui.app import _single_table, _single_table_uri, _upgrade_refused
from strata.notebook.tui.client import TuiClient, TuiClientError
from strata.notebook.tui.viewmodel import NotebookViewModel
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


@pytest.fixture
def opened(notebook_personal_server):
    # The server opens only notebooks under its storage dir.
    storage = Path(notebook_personal_server["config"].notebook_storage_dir)
    storage.mkdir(parents=True, exist_ok=True)
    nb = create_notebook(storage / "nb", "Spectated", initialize_environment=False)
    add_cell_to_notebook(nb, "t", None)
    # It displays the table it stores: a resync carries displays, not variables.
    write_cell(nb, "t", "import pyarrow as pa\ntable = pa.table({'n': [3, 1, 2]})\ntable\n")
    add_cell_to_notebook(nb, "r", "t")
    write_cell(nb, "r", "rows = table.num_rows\n")
    return TuiClient(notebook_personal_server["base_url"]), nb


async def test_opening_by_path_gives_the_session_the_picker_lists(opened):
    client, nb = opened

    session_id = (await client.open_notebook(str(nb)))["session_id"]
    sessions = await client.list_sessions()

    (listed,) = [s for s in sessions if s.get("session_id") == session_id]
    # What the picker shows, and what a reopen by path uses.
    assert listed["name"] == "Spectated"
    assert Path(listed["path"]).resolve() == nb.resolve()
    # A second open of the path reuses the session.
    assert (await client.open_notebook(str(nb)))["session_id"] == session_id


async def _state_over_the_socket(client: TuiClient, session_id: str) -> NotebookViewModel:
    """What the TUI does on connect: send its sync, apply the ``notebook_state`` reply."""
    async with websockets.connect(client.ws_url(session_id), max_size=None) as ws:
        await ws.send(json.dumps({"type": "notebook_sync", "seq": 0, "payload": {}}))
        while True:
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            if message["type"] == "notebook_state":
                vm = NotebookViewModel()
                vm.apply_notebook_state(message["payload"])
                return vm


async def test_the_socket_answers_the_tuis_sync_with_the_notebook(opened):
    client, nb = opened
    session_id = (await client.open_notebook(str(nb)))["session_id"]

    vm = await _state_over_the_socket(client, session_id)

    assert vm.cell_order == ["t", "r"]


async def test_an_unknown_session_is_refused_the_way_the_tui_stops_on(opened):
    client, _nb = opened

    with pytest.raises(websockets.exceptions.InvalidStatus) as refused:
        async with websockets.connect(client.ws_url("no-such-session")):
            pass

    # A 403, which the TUI reads as "gone" rather than reconnecting forever.
    assert _upgrade_refused(refused.value)


async def test_the_table_the_viewer_finds_pages_from_the_server(opened):
    client, nb = opened
    session_id = (await client.open_notebook(str(nb)))["session_id"]
    async with httpx.AsyncClient(base_url=client.server_url, timeout=120) as http:
        run = await http.post(f"/v1/notebooks/{session_id}/cells/t/execute")
    assert run.status_code == 200 and run.json()["status"] == "ready", run.text

    # The URI the viewer pages is the one on the cell's output, as the TUI reads it.
    cell = (await _state_over_the_socket(client, session_id)).cells["t"]
    table, uri = _single_table(cell), _single_table_uri(cell)
    assert table is not None and uri is not None
    assert table[0] == ["n"]

    page = await client.get_cell_data_page(session_id, "t", uri, limit=2, sort_by="n")

    assert page["pageable"] is True
    assert (page["total"], page["offset"]) == (3, 0)
    assert [row[0] for row in page["rows"]] == [1, 2]
    assert page["columns"] == ["n"]


async def test_a_refusal_carries_the_servers_reason(opened):
    client, nb = opened
    session_id = (await client.open_notebook(str(nb)))["session_id"]

    with pytest.raises(TuiClientError, match=r"^server returned 400: Malformed artifact_uri$"):
        await client.get_cell_data_page(session_id, "t", "not-a-uri")


async def test_an_unreachable_server_says_so(unused_tcp_port):
    client = TuiClient(f"http://127.0.0.1:{unused_tcp_port}")

    with pytest.raises(TuiClientError, match="^cannot reach http://127.0.0.1:"):
        await client.list_sessions()
