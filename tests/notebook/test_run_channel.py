"""The channel a cell publishes checkpoints through, inside the cell's process."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from strata.notebook import run_channel, serializer
from strata.notebook.notebook_client import StrataClient
from strata.notebook.run_channel import RunChannel, from_pipe, is_event_line


def _channel(run_dir: Path, lines: list[str], restore: dict | None = None) -> RunChannel:
    return RunChannel(
        lines.append,
        run_dir,
        serialize=serializer.serialize_value,
        deserialize=serializer.deserialize_value,
        restore=restore,
    )


def test_a_checkpoint_writes_its_parts_and_emits_one_line(tmp_path: Path):
    lines: list[str] = []
    channel = _channel(tmp_path, lines)

    assert channel.checkpoint(model={"w": 1}, metrics=[0.5]) == 1
    assert channel.checkpoint(model={"w": 2}) == 2

    assert len(lines) == 2 and all(line.endswith("\n") for line in lines)
    first = json.loads(lines[0])
    assert first["event"] == "checkpoint" and first["sequence"] == 1
    assert set(first["parts"]) == {"model", "metrics"}
    spec = first["parts"]["model"]
    assert spec["file"].startswith("checkpoints/1/") and (tmp_path / spec["file"]).is_file()
    assert serializer.deserialize_value(spec["content_type"], tmp_path / spec["file"]) == {"w": 1}
    assert is_event_line(lines[0]) and not is_event_line('{"success": true}')


def test_a_checkpoint_needs_a_part(tmp_path: Path):
    with pytest.raises(ValueError, match="named part"):
        _channel(tmp_path, []).checkpoint()


def test_restore_returns_the_manifest_parts_and_continues_the_sequence(tmp_path: Path):
    (tmp_path / "restore").mkdir()
    spec = serializer.serialize_value({"epoch": 3}, tmp_path / "restore", "metrics")
    restore = {
        "release_id": "r1",
        "sequence": 4,
        "parts": {
            "metrics": {"content_type": spec["content_type"], "file": f"restore/{spec['file']}"}
        },
    }
    lines: list[str] = []
    channel = _channel(tmp_path, lines, restore)

    assert channel.restore() == {"metrics": {"epoch": 3}}
    assert channel.restore() is channel.restore()  # read once
    assert channel.checkpoint(metrics={"epoch": 4}) == 5
    assert _channel(tmp_path, []).restore() is None


def test_a_head_names_a_checkpoint_by_sequence(tmp_path: Path):
    lines: list[str] = []
    channel = _channel(tmp_path, lines)
    channel.head("best-validation", 2)
    assert json.loads(lines[0]) == {"event": "head", "name": "best-validation", "sequence": 2}
    with pytest.raises(ValueError):
        channel.head("a/b", 1)
    with pytest.raises(ValueError):
        channel.head("best", 0)
    with pytest.raises(ValueError):
        channel.head("best", True)


def test_the_client_delegates_to_the_channel_or_says_there_is_none(tmp_path: Path):
    lines: list[str] = []
    with_channel = StrataClient("http://localhost:1", channel=_channel(tmp_path, lines))
    assert with_channel.checkpoint(x=1) == 1
    assert with_channel.restore() is None
    with_channel.head("best", 1)
    assert [json.loads(line)["event"] for line in lines] == ["checkpoint", "head"]

    without = StrataClient("http://localhost:1")
    for call in (lambda: without.checkpoint(x=1), without.restore, lambda: without.head("b", 1)):
        with pytest.raises(RuntimeError, match="no checkpoint channel"):
            call()


@pytest.mark.skipif(sys.platform == "win32", reason="no descriptor passing on Windows")
def test_the_pipe_channel_writes_lines_the_executor_reads(tmp_path: Path, monkeypatch):
    read_fd, write_fd = os.pipe()
    monkeypatch.setenv(run_channel.EVENTS_FD_ENV, str(write_fd))
    channel = from_pipe(
        {"output_dir": str(tmp_path)},
        serialize=serializer.serialize_value,
        deserialize=serializer.deserialize_value,
    )
    assert channel is not None
    assert not os.get_inheritable(write_fd), "a cell's children must not hold the pipe open"
    channel.checkpoint(x=1)
    channel.close()
    with os.fdopen(read_fd) as reader:
        assert [json.loads(line)["event"] for line in reader] == ["checkpoint"]

    monkeypatch.delenv(run_channel.EVENTS_FD_ENV)
    assert (
        from_pipe(
            {"output_dir": str(tmp_path)},
            serialize=serializer.serialize_value,
            deserialize=serializer.deserialize_value,
        )
        is None
    )
