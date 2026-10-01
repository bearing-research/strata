"""Driver adapter registry, keyed by name (matched against ``ConnectionSpec.driver``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strata.notebook.sql.adapter import DriverAdapter


_REGISTRY: dict[str, DriverAdapter] = {}


def register_adapter(adapter: DriverAdapter) -> None:
    """Register a ``DriverAdapter`` under its ``name``; re-registration replaces it."""
    _REGISTRY[adapter.name] = adapter


def get_adapter(name: str) -> DriverAdapter:
    """Look up the adapter registered for ``name``.

    Raises ``KeyError`` listing the known drivers, which the executor shows as
    "unknown driver".
    """
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(
            f"unknown SQL driver: {name!r}. Known drivers: {known}",
        )
    return _REGISTRY[name]


def known_drivers() -> list[str]:
    """List currently registered driver names, sorted."""
    return sorted(_REGISTRY)


def _reset_for_tests() -> None:
    """Drop all registrations (test-only); restore with ``_restore_defaults_for_tests``."""
    _REGISTRY.clear()


def _restore_defaults_for_tests() -> None:
    """Re-register the built-in adapters; pairs with ``_reset_for_tests``."""
    _REGISTRY.clear()
    from strata.notebook.sql.drivers import register_default_adapters

    register_default_adapters()
