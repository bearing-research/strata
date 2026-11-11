"""Shared pieces of the warm-pool tests."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from tests.conftest import LIVE_SERVER_TIMEOUT

if TYPE_CHECKING:
    from strata.notebook.pool import WarmProcessPool


async def warm_again(pool: WarmProcessPool) -> None:
    """Wait until the pool's worker is ready; a used one's replacement starts in the background.

    A cell run before then finds the pool empty and quietly runs cold.
    """
    deadline = time.monotonic() + LIVE_SERVER_TIMEOUT
    while pool._available.qsize() < pool.pool_size:
        assert time.monotonic() < deadline, "the warm worker never signalled ready"
        await asyncio.sleep(0.05)
