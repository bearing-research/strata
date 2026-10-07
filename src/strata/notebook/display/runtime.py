"""Runtime helpers for notebook display side effects.

``harness.py`` and ``pool_worker.py`` load this file by path, not as a package,
so it must have no relative imports (they would break those loaders silently).
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class Markdown:
    """Explicit markdown display wrapper for notebook cells."""

    text: str

    def _repr_markdown_(self) -> str:
        return self.text

    def __str__(self) -> str:
        return self.text


# Callers exclude these from mutation fingerprinting (``display`` accumulates
# captured values, so it "changes" every run) and clear them between batch cells.
DISPLAY_HELPER_NAMES = ("display", "Markdown")


class DisplayCapture:
    """Capture explicit display side effects during cell execution, in order."""

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
        # Side-effecting, like IPython.display.display(); no separate value.
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
        """Capture ``plt.show()`` and ``Figure.show()`` as display outputs.

        Imports nothing itself: pyplot is patched now if loaded, else when the cell
        imports it. Importing it for every cell costs time and, on a fresh install,
        prints matplotlib's font-cache notice into whichever cell ran first.
        """
        restore: list[tuple[Any, str, Any]] = []

        def _capture_current_figures(plt: Any) -> None:
            try:
                figure_numbers = list(plt.get_fignums())
            except Exception:
                return
            for number in figure_numbers:
                try:
                    self.capture(plt.figure(number))
                except Exception:
                    continue

        def _patch(plt: Any) -> None:
            figure_cls = getattr(sys.modules.get("matplotlib.figure"), "Figure", None)
            original_show = getattr(plt, "show", None)
            original_figure_show = getattr(figure_cls, "show", None) if figure_cls else None

            def _patched_show(*_args: Any, **_kwargs: Any) -> None:
                _capture_current_figures(plt)
                return None

            def _patched_figure_show(fig_self: Any, *_args: Any, **_kwargs: Any) -> None:
                self.capture(fig_self)
                return None

            # ``setattr`` bypasses static typing: the patched callables have looser
            # signatures than ``plt.show`` / ``Figure.show``.
            if callable(original_show):
                setattr(plt, "show", _patched_show)
                restore.append((plt, "show", original_show))
            if figure_cls is not None and callable(original_figure_show):
                setattr(figure_cls, "show", _patched_figure_show)
                restore.append((figure_cls, "show", original_figure_show))

        hook: _AfterImport | None = None
        if "matplotlib.pyplot" in sys.modules:
            _patch(sys.modules["matplotlib.pyplot"])
        else:
            hook = _AfterImport("matplotlib.pyplot", _patch)
            sys.meta_path.insert(0, hook)

        try:
            yield
        finally:
            if hook is not None and hook in sys.meta_path:
                sys.meta_path.remove(hook)
            for owner, attr, original in reversed(restore):
                setattr(owner, attr, original)


class _AfterImport:
    """A one-shot ``sys.meta_path`` finder that runs *callback* on *name* once it is imported."""

    def __init__(self, name: str, callback: Callable[[ModuleType], None]) -> None:
        self.name = name
        self.callback = callback

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> ModuleSpec | None:
        if fullname != self.name:
            return None
        # Out of the way first, so the real lookup below does not find this finder again.
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        loader = spec.loader if spec is not None else None
        if loader is None or not hasattr(loader, "exec_module"):
            return spec
        exec_module = loader.exec_module

        def _exec_then_callback(module: ModuleType) -> None:
            exec_module(module)
            self.callback(module)

        setattr(loader, "exec_module", _exec_then_callback)
        return spec
