"""Who has a notebook session open, which cell each is on, and soft edit locks.

Every socket on a session receives every frame, and ``cell_source_update`` was
last flush wins: two people editing one cell overwrote each other without
either knowing. This module keeps two small tables on the session.

**Presence.** One entry per identity: the principal under ``trusted_proxy`` or
``api_key``, otherwise the author a client declares (``strata agent`` and MCP
clients name themselves) and ``local`` when it declares none. Each socket
contributes its identity and the cell it last focused; an API edit (an agent
driving the notebook over REST, which has no socket) contributes its author on
the cell it edited, for ``API_PRESENCE_SECONDS``. Two tabs of one person are
one entry.

**Soft locks.** The identity that last changed a cell holds it for
``notebook_cell_lock_seconds``. A different identity's edit inside that window
is refused with the holder's name unless it says ``force``. One person, however
many tabs, never contends with themselves, so a single-user session behaves as
it did before.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_LOCK_SECONDS = 5.0
# How long an edit made without a socket keeps its author in presence. There is
# no disconnect to end it, so it ends by going quiet.
API_PRESENCE_SECONDS = 60.0


@dataclass
class _Seat:
    principal: str
    focused_cell_id: str | None
    since: float  # wall clock, for display
    touched: float  # monotonic, for ordering and expiry


class SessionPresence:
    """The presence and lock tables of one notebook session."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ):
        self._clock = clock
        self._wall = wall
        self._sockets: dict[object, _Seat] = {}
        self._api: dict[str, _Seat] = {}
        self._editors: dict[str, tuple[str, float]] = {}

    # -- presence -----------------------------------------------------------

    def join(self, key: object, principal: str) -> None:
        now = self._clock()
        self._sockets[key] = _Seat(principal, None, self._wall(), now)

    def principal_of(self, key: object) -> str | None:
        seat = self._sockets.get(key)
        return seat.principal if seat is not None else None

    def leave(self, key: object) -> bool:
        return self._sockets.pop(key, None) is not None

    def focus(self, key: object, principal: str, cell_id: str | None) -> bool:
        """Record that the socket *key* is on *cell_id*; return whether that
        changed what presence shows."""
        seat = self._sockets.get(key)
        if seat is None:
            return False
        if seat.principal == principal and seat.focused_cell_id == cell_id:
            return False
        self._sockets[key] = _Seat(principal, cell_id, self._wall(), self._clock())
        return True

    def api_edit(self, principal: str, cell_id: str) -> None:
        self._api[principal] = _Seat(principal, cell_id, self._wall(), self._clock())

    def snapshot(self) -> list[dict[str, object]]:
        """``[{principal, focused_cell_id, since}]``, one per identity, the
        most recent focus winning when an identity has several seats."""
        now = self._clock()
        for principal, seat in list(self._api.items()):
            if now - seat.touched >= API_PRESENCE_SECONDS:
                del self._api[principal]
        latest: dict[str, _Seat] = {}
        for seat in [*self._sockets.values(), *self._api.values()]:
            held = latest.get(seat.principal)
            if held is None or seat.touched >= held.touched:
                latest[seat.principal] = seat
        return [
            {
                "principal": seat.principal,
                "focused_cell_id": seat.focused_cell_id,
                "since": seat.since,
            }
            for seat in sorted(latest.values(), key=lambda s: s.principal)
        ]

    # -- soft locks ---------------------------------------------------------

    def holder(self, cell_id: str, principal: str, window_seconds: float) -> str | None:
        """Who else changed *cell_id* within the window, if anyone."""
        editor = self._editors.get(cell_id)
        if editor is None or editor[0] == principal:
            return None
        if self._clock() - editor[1] >= window_seconds:
            return None
        return editor[0]

    def record_edit(self, cell_id: str, principal: str) -> None:
        self._editors[cell_id] = (principal, self._clock())


def lock_window_seconds() -> float:
    """``notebook_cell_lock_seconds`` from the server's config."""
    try:
        from strata.server import get_state

        config = get_state().config
    except RuntimeError:
        return DEFAULT_LOCK_SECONDS
    return float(getattr(config, "notebook_cell_lock_seconds", DEFAULT_LOCK_SECONDS))
