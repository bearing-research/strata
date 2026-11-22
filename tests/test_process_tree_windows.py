"""Stopping a process on Windows, where the graceful signal is a console Ctrl-Break.

The POSIX side is ``tests/notebook/test_process_tree.py``.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from strata.notebook.process_tree import (
    subprocess_kwargs_for_new_group,
    terminate_subprocess_tree,
)

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Ctrl-Break is Windows's")

# As a venv's interpreter that was not up for the first Ctrl-Break: it goes on the second.
_MISSES_THE_FIRST = """
import signal, sys, time
seen = []
def on_break(signum, frame):
    seen.append(signum)
    if len(seen) == 2:
        sys.exit(7)
signal.signal(signal.SIGBREAK, on_break)
print("ready", flush=True)
while True:
    time.sleep(0.02)
"""


async def test_a_process_that_missed_the_ctrl_break_is_sent_it_again():
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _MISSES_THE_FIRST,
        stdout=asyncio.subprocess.PIPE,
        **subprocess_kwargs_for_new_group(),
    )
    assert proc.stdout is not None
    assert await asyncio.wait_for(proc.stdout.readline(), timeout=30) == b"ready\r\n"

    await terminate_subprocess_tree(proc)

    assert proc.returncode == 7, "killed, not stopped by the second Ctrl-Break"
