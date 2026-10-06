"""Built-in SQL driver adapters.

Each module exposes ``register()``, which ``register_default_adapters`` calls.
A driver module must import its ADBC package lazily (inside ``open()``), so the
module always imports and registers, and ``open()`` raises a ``RuntimeError``
with the install hint. ``ImportError`` is therefore never swallowed here.
"""

from __future__ import annotations

import importlib

# Add an entry only once the module exists and exposes ``register()``.
_BUILTIN_DRIVERS: tuple[str, ...] = (
    "postgresql",
    "sqlite",
    "snowflake",
    "bigquery",
    "duckdb",
)


def register_default_adapters() -> None:
    """Import each built-in driver module and call its ``register()``.

    Explicit, not import-time registration, because a cached module does not
    re-run on import and a registry reset by ``_reset_for_tests`` would stay
    empty. Idempotent. ``ImportError`` from a built-in module propagates as a bug.
    """
    for module_name in _BUILTIN_DRIVERS:
        mod = importlib.import_module(f"strata.notebook.sql.drivers.{module_name}")
        register_fn = getattr(mod, "register", None)
        if not callable(register_fn):
            raise RuntimeError(
                f"built-in driver module {module_name!r} is missing a "
                "callable ``register()``; every driver module must expose one"
            )
        register_fn()


def builtin_driver_names() -> tuple[str, ...]:
    """Names of the built-in driver modules ``register_default_adapters`` imports."""
    return _BUILTIN_DRIVERS
