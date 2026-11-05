"""Print ``str(value)`` of one stored object, as JSON, for a prompt cell that reads it.

Runs in the notebook's venv as cell code, because loading a pickle runs its code and the
server never does. Loads the sibling ``serializer.py`` by path instead of importing
``strata``. Usage: ``object_text_harness.py <content_type> <path> <max_chars>``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_serializer():
    path = Path(__file__).parent / "serializer.py"
    spec = importlib.util.spec_from_file_location("_nb_serializer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered first: dataclass and pickle lookups go through sys.modules.
    sys.modules["_nb_serializer"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    content_type, path, max_chars = sys.argv[1], sys.argv[2], int(sys.argv[3])
    # A ``__str__`` that prints must not corrupt the one line the server parses.
    result_stream, sys.stdout = sys.stdout, sys.stderr
    try:
        value = _load_serializer().deserialize_value(content_type, Path(path))
        result = {"type": type(value).__name__, "text": str(value)[:max_chars]}
    except Exception as exc:  # noqa: BLE001 - whatever the object raises is the answer
        result = {"error": f"{type(exc).__name__}: {exc}"}
    result_stream.write(json.dumps(result))


if __name__ == "__main__":
    main()
