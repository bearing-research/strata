"""Runtime helpers for notebook display-side effects.

This module is intentionally dependency-light so it can be imported both by
the server package *and* by notebook subprocess helpers via direct file
loading (``harness.py`` and ``pool_worker.py`` use
``importlib.util.spec_from_file_location`` to load this file without setting
up ``strata.notebook.display`` as a package). That load path imposes one
invariant: **no relative imports in this module**. Absolute imports (stdlib,
third-party) are fine; ``from .other import X`` would break the subprocess
loaders silently.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Markdown:
    """Explicit markdown display wrapper for notebook cells."""

    text: str

    def _repr_markdown_(self) -> str:
        return self.text

    def __str__(self) -> str:
        return self.text


# Names ``DisplayCapture.install`` injects into the cell namespace. Callers
# exclude these from mutation fingerprinting (the ``display`` helper accumulates
# captured values, so it "changes" every run) and clear them between batch cells.
DISPLAY_HELPER_NAMES = ("display", "Markdown")


class DisplayCapture:
    """Capture explicit display-side effects during cell execution.

    Visible outputs are captured in order. A legacy last-item compatibility
    shim is handled by higher layers.
    """

    def __init__(self) -> None:
        self._values: list[Any] = []

    def capture(self, value: Any) -> Any:
        """Record *value* as a visible display output and return it."""
        if value is not None:
            self._values.append(value)
        return value

    def display(self, value: Any) -> Any:
        """Notebook-visible display helper injected into cell globals."""
        self.capture(value)
        # Mirror notebook display helpers like IPython.display.display(),
        # which are side-effecting and do not produce a separate value.
        return None

    def install(self, namespace: dict[str, Any]) -> None:
        """Inject display helpers into the execution namespace."""
        namespace.setdefault("display", self.display)
        namespace.setdefault("Markdown", Markdown)

    def resolve(self, last_expression_value: Any | None) -> list[Any]:
        """Return ordered visible outputs after one cell execution."""
        if last_expression_value is not None:
            self.capture(last_expression_value)
        return list(self._values)

    @contextmanager
    def capture_side_effects(self):
        """Capture common notebook-side display effects like ``plt.show()``."""
        plt = None
        figure_cls = None
        original_show = None
        original_figure_show = None

        try:
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure as figure_cls
        except ImportError:
            plt = None  # type: ignore[assignment]

        if plt is None:
            yield
            return

        original_show = getattr(plt, "show", None)
        original_figure_show = getattr(figure_cls, "show", None) if figure_cls else None

        def _capture_current_figures() -> None:
            try:
                figure_numbers = list(plt.get_fignums())
            except Exception:
                return
            for number in figure_numbers:
                try:
                    self.capture(plt.figure(number))
                except Exception:
                    continue

        def _patched_show(*_args: Any, **_kwargs: Any) -> None:
            _capture_current_figures()
            return None

        def _patched_figure_show(fig_self: Any, *_args: Any, **_kwargs: Any) -> None:
            self.capture(fig_self)
            return None

        # ``setattr`` deliberately bypasses static attribute typing — the
        # patched callables intentionally have looser signatures than
        # ``plt.show`` / ``Figure.show`` (we ignore their kwargs to
        # capture the figure for display).
        if callable(original_show):
            setattr(plt, "show", _patched_show)
        if figure_cls is not None and callable(original_figure_show):
            setattr(figure_cls, "show", _patched_figure_show)

        try:
            yield
        finally:
            if callable(original_show):
                setattr(plt, "show", original_show)
            if figure_cls is not None and callable(original_figure_show):
                setattr(figure_cls, "show", original_figure_show)
