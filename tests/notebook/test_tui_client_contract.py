"""The TUI's REST client against a real server: the fields the spectator reads are there.

The app tests replace the client with fakes shaped like the responses, so a route that
renamed a field would pass them and break the TUI.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import websockets

from strata.notebook.tui.client import TuiClient, TuiClientError
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


@pytest.fixture
def opened(notebook_personal_server):
    # The server opens only notebooks under its storage dir.
    storage = Path(notebook_personal_server["config"].notebook_storage_dir)
    storage.mkdir(parents=True, exist_ok=True)
    nb = create_notebook(storage / "nb", "Spectated", initialize_environment=False)
    add_cell_to_notebook(nb, "t", None)
    write_cell(nb, "t", "import pyarrow as pa\ntable = pa.table({'n': [3, 1, 2]})\n")
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


async def test_the_websocket_url_is_one_the_server_accepts(opened):
    client, nb = opened
    session_id = (await client.open_notebook(str(nb)))["session_id"]

    async with websockets.connect(client.ws_url(session_id), max_size=None) as ws:
        assert ws.protocol.state.name == "OPEN"


async def test_a_page_of_a_table_output_is_what_the_viewer_renders(opened):
    client, nb = opened
    session_id = (await client.open_notebook(str(nb)))["session_id"]
    base = client.server_url
    run = httpx.post(f"{base}/v1/notebooks/{session_id}/cells/t/execute", timeout=120)
    assert run.status_code == 200, run.text
    state = httpx.get(f"{base}/v1/notebooks/sessions/{session_id}", timeout=30).json()
    uri = next(c for c in state["cells"] if c["id"] == "t")["artifact_uris"]["table"]

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
