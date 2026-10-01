"""Built-in SQL driver adapters.

Each adapter module exposes a ``register()`` callable that adds its
``DriverAdapter`` to the global registry. ``register_default_adapters``
imports each built-in module and calls ``register()``.

**Driver-module convention**: a built-in driver module MUST keep its
ADBC package import lazy (inside ``open()`` or behind a guard) so
importing the module always succeeds, even when the optional ADBC
package isn't installed. The adapter still registers; ``open()``
raises a clear ``RuntimeError`` at execute time with the install hint.

This convention means we don't need to swallow ``ImportError`` here —
a real bug in a driver module surfaces immediately instead of being
masked as "driver unavailable."
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

    Calling ``register()`` explicitly (rather than relying on
    module-level side effects) is necessary because Python caches
    imported modules: a second import doesn't re-execute the module
    body, so a registry that was reset via ``_reset_for_tests`` would
    stay empty if we only relied on import-time registration.

    Idempotent. ``ImportError`` from a built-in driver module
    propagates — that means a real bug, not a missing optional
    package.
    """
    for module_name in _BUILTIN_DRIVERS:
        mod = importlib.import_module(f"strata.notebook.sql.drivers.{module_name}")
        register_fn = getattr(mod, "register", None)
        if not callable(register_fn):
            raise RuntimeError(
                f"built-in driver module {module_name!r} is missing a "
                "callable ``register()`` — every driver module must expose one"
            )
        register_fn()


def builtin_driver_names() -> tuple[str, ...]:
    """List of built-in driver module names ``register_default_adapters``
    will import. Used by tests to verify every advertised driver
    actually exists and registers.
    """
    return _BUILTIN_DRIVERS
