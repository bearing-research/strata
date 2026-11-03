"""A session behind a host that remembers what it was asked for."""

from __future__ import annotations

from typing import Any


class RecordingHost:
    """A session that remembers any use the ``ExecutorHost`` protocol does not list.

    It passes everything through and raises nothing: the executor's broad excepts would turn
    a raise into a failed cell, and the trace would report that instead of the attribute.
    """

    def __init__(self, session: Any, members: set[str], strays: set[str]) -> None:
        object.__setattr__(self, "_host", (session, members, strays))

    def __getattr__(self, name: str) -> Any:
        session, members, strays = object.__getattribute__(self, "_host")
        if name not in members:
            strays.add(f"read {name}")
        return getattr(session, name)

    def __setattr__(self, name: str, value: Any) -> None:
        session, _, strays = object.__getattribute__(self, "_host")
        strays.add(f"set {name}")
        setattr(session, name, value)


def host_members() -> set[str]:
    from strata.notebook.executor_host import ExecutorHost

    return {name for name in vars(ExecutorHost) if not name.startswith("_")}
