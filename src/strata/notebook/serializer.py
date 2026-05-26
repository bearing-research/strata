"""Shared serialization/deserialization for notebook cell values.

Supports these content types:
  arrow/ipc    — Anything Arrow-representable (PyArrow Tables/RecordBatch,
                 pandas DataFrames/Series, numpy ndarrays of any dim,
                 numpy scalars, typed Python primitives like datetime /
                 Decimal / UUID / bytes / complex). Shape is encoded in
                 schema metadata: ``strata.arrow.shape`` = "table" |
                 "tensor" | "scalar".
  json/object  — dicts, lists, scalars (int/float/str/bool/None)
  image/png    — Displayable PNG output (figures, images)
  text/markdown — Displayable markdown output
  module/import — Python module objects (re-imported by name on read)
  module/cell  — Synthetic module export for top-level defs/classes
  module/cell-instance — Instance of a synthetic notebook-exported class
  pickle/object — everything else
  application/x-r-rds — R-only RDS blob produced by harness.R; refuses to
                 deserialize from Python (raises ``StrataRArtifactError``).

This module is loaded dynamically by harness.py, pool_worker.py, and
inspect_repl.py via ``importlib.util``, since those scripts run inside
the notebook's own venv and cannot ``import strata``.

Loading pattern (used in each subprocess script):

    import importlib.util as _ilu
    from pathlib import Path as _Path

    def _load_serializer():
        _p = _Path(__file__).parent / "serializer.py"
        _spec = _ilu.spec_from_file_location("_nb_serializer", _p)
        _m = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        return _m

    _ser = _load_serializer()
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import pickle
import sys
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any, NamedTuple, NotRequired, Protocol, TypedDict

logger = logging.getLogger(__name__)


class ContentType(StrEnum):
    """Content type strings used by the notebook serializer.

    StrEnum rather than bare strings so call sites get autocomplete,
    typos become import-time errors, and the complete set is
    inventoried here. StrEnum values remain plain ``str``, so existing
    code comparing against ``"arrow/ipc"`` continues to interop
    cleanly during migration.

    Defined inside serializer.py (not a sibling module) because this
    file is loaded via ``importlib.util`` inside the notebook venv,
    which doesn't have ``strata`` importable.
    """

    ARROW_IPC = "arrow/ipc"
    JSON_OBJECT = "json/object"
    PICKLE_OBJECT = "pickle/object"
    IMAGE_PNG = "image/png"
    TEXT_MARKDOWN = "text/markdown"
    MODULE_IMPORT = "module/import"
    MODULE_CELL = "module/cell"
    MODULE_CELL_INSTANCE = "module/cell-instance"
    # R-only payload — saveRDS() blob produced by harness.R for any
    # value that isn't a data.frame/tibble (Arrow tier) or
    # atomic/list (JSON tier). The bytes are unreadable from Python;
    # ``_deserialize_rds`` raises ``StrataRArtifactError`` to tell
    # the user to re-export from R as a data.frame for cross-language
    # handoff.
    RDS_OBJECT = "application/x-r-rds"


class StrataRArtifactError(RuntimeError):
    """Raised when a Python cell tries to consume an R-only RDS artifact.

    Attached attributes give callers the context to render a useful
    error to the user without re-parsing the message string:

    ``code``         — stable identifier ``R_ONLY_ARTIFACT``
    ``file_path``    — on-disk path of the RDS blob (debugging only)
    ``variable_name``— upstream variable, when known. Populated by the
                       Python harness when it catches the error during
                       input deserialization.
    """

    code = "R_ONLY_ARTIFACT"

    def __init__(
        self,
        file_path: Path | str,
        *,
        variable_name: str | None = None,
        message: str | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.variable_name = variable_name
        if message is None:
            target = (
                f"variable '{variable_name}'" if variable_name else f"artifact at {self.file_path}"
            )
            message = (
                f"Cannot consume R-only artifact ({target}) from Python: "
                "the upstream R cell stored this value via saveRDS(), which "
                "Python cannot read. Re-export the upstream as a data.frame "
                "or tibble (handed across as Arrow IPC) for cross-language "
                "consumption."
            )
        super().__init__(message)


class SerializedPayload(TypedDict):
    """Metadata dict returned by ``serialize_value`` and every ``_serialize_*`` helper.

    Four keys are always present (``content_type``, ``file``, ``bytes``,
    ``preview``); the rest are content-type-specific extras. Keeping
    them all in one ``TypedDict`` instead of per-handler subclasses
    matches the actual on-wire shape and keeps typo protection at
    every literal-dict construction site.
    """

    content_type: ContentType
    file: str
    bytes: int
    preview: Any
    # arrow/ipc table shape
    rows: NotRequired[int]
    columns: NotRequired[list[Any]]
    # text/markdown
    markdown_text: NotRequired[str]
    # image/png
    inline_data_url: NotRequired[str]
    width: NotRequired[int | None]
    height: NotRequired[int | None]
    # pickle/object & module/cell-instance
    codec: NotRequired[str]
    type: NotRequired[str]


OBJECT_CODEC_ENV_VAR = "STRATA_NOTEBOOK_OBJECT_CODEC"
_CODEC_ENVELOPE_TAG = "strata.notebook.object_codec.v1"
_CELL_INSTANCE_STATE_TAG = "strata.notebook.cell_instance_state.v1"
_ARROW_JSON_FALLBACK_TAG = "strata.notebook.arrow_json_fallback.v1"

# Wire-stable envelope keys. These are byte-level constants that
# already-cached artifacts depend on — changing any value invalidates
# every prior pickle/JSON-fallback/instance-state artifact on disk.
_TAG_OBJECT_CODEC = "__strata_object_codec__"
_TAG_ARROW_JSON_FALLBACK = "__strata_arrow_json_fallback__"
_TAG_CELL_INSTANCE_STATE = "__strata_cell_instance_state__"

# Module-level attributes that mark a synthetic cell-exported module
# and its exported classes. Looked up via getattr/setattr on the
# module dict and the class itself.
_CELL_MODULE_SOURCE_ATTR = "__strata_cell_module_source__"
_CELL_MODULE_FLAG_ATTR = "__strata_cell_module__"
_CELL_EXPORTED_CLASS_ATTR = "__strata_cell_exported_class__"


class ObjectCodec(Protocol):
    """Pluggable object serializer backend for notebook runtime values."""

    name: str

    def dumps(self, value: Any) -> bytes:
        """Serialize *value* to backend-specific bytes."""

    def loads(self, data: bytes) -> Any:
        """Deserialize backend-specific bytes to a Python object."""


class _PickleObjectCodec:
    name = "pickle"

    def dumps(self, value: Any) -> bytes:
        return pickle.dumps(value, protocol=5)

    def loads(self, data: bytes) -> Any:
        return pickle.loads(data)


class _CloudPickleObjectCodec:
    name = "cloudpickle"

    def __init__(self) -> None:
        try:
            import cloudpickle
        except ImportError as exc:  # pragma: no cover - optional backend
            raise ValueError(
                "Object codec 'cloudpickle' requires the 'cloudpickle' package to be installed"
            ) from exc
        self._cloudpickle = cloudpickle

    def dumps(self, value: Any) -> bytes:
        return self._cloudpickle.dumps(value, protocol=5)

    def loads(self, data: bytes) -> Any:
        return pickle.loads(data)


def _resolve_object_codec(codec_name: str | None = None) -> ObjectCodec:
    """Return the configured object codec implementation.

    Default is cloudpickle — it's a strict superset of stdlib pickle
    (handles lambdas, closures, nested classes, dynamically-defined
    functions) and ships with the ``notebook`` extra. Users can opt
    out by setting the env var to ``pickle``. If cloudpickle can't be
    imported (e.g. someone installed only core ``strata`` and still
    spun up the notebook runtime), we transparently fall back to
    stdlib pickle so cells don't fail to serialize.
    """
    selected = (codec_name or os.environ.get(OBJECT_CODEC_ENV_VAR, "cloudpickle")).strip().lower()
    if selected == "cloudpickle":
        try:
            return _CloudPickleObjectCodec()
        except ValueError:
            return _PickleObjectCodec()
    if selected == "pickle":
        return _PickleObjectCodec()
    raise ValueError(
        f"Unknown notebook object codec '{selected}'. Supported codecs: pickle, cloudpickle"
    )


def _wrap_codec_payload(codec_name: str, payload: bytes) -> dict[str, Any]:
    return {
        _TAG_OBJECT_CODEC: _CODEC_ENVELOPE_TAG,
        "codec": codec_name,
        "payload": payload,
    }


def _unwrap_codec_payload(obj: Any) -> tuple[str, bytes] | None:
    if not isinstance(obj, dict):
        return None
    if obj.get(_TAG_OBJECT_CODEC) != _CODEC_ENVELOPE_TAG:
        return None
    codec_name = obj.get("codec")
    payload = obj.get("payload")
    if not isinstance(codec_name, str) or not isinstance(payload, bytes):
        raise ValueError("Invalid notebook object codec envelope")
    return codec_name, payload


# ---------------------------------------------------------------------------
# Content-type detection
# ---------------------------------------------------------------------------


def detect_content_type(value: Any, variable_name: str | None = None) -> ContentType:
    """Return the content type for *value*.

    Called from the notebook subprocess (harness / pool_worker), which
    runs in a venv with pyarrow (core dep) and pandas/numpy (notebook
    extra) installed. Imports stay lazy inside the function so that
    loading this module doesn't pay ~400ms of pyarrow+pandas+numpy init
    cost when detection never fires.

    Detection order (first match wins):
      1. Anything Arrow-representable → arrow/ipc
         (pyarrow Table/RecordBatch, pandas DataFrame/Series, numpy
         ndarray of any dim, numpy scalars, typed Python primitives
         like datetime/Decimal/UUID/bytes/complex)
      2. Markdown / PNG display value → text/markdown or image/png
      3. JSON-serializable primitive  → json/object
      4. Python module                → module/import
      5. Cell-defined class instance  → module/cell-instance
      6. Anything else                → pickle/object (fallback)
    """
    import types

    if _is_arrow_representable(value):
        return ContentType.ARROW_IPC

    if _is_display_variable_name(variable_name):
        if _is_markdown_display_value(value):
            return ContentType.TEXT_MARKDOWN
        if _is_png_display_value(value):
            return ContentType.IMAGE_PNG

    if isinstance(value, (dict, list, int, float, str, bool, type(None))):
        # Typed structurally JSON-safe but could still contain NaN, Inf,
        # non-string dict keys, or nested non-primitives. A probe write
        # confirms the value survives the actual JSON writer.
        try:
            json.dumps(value)
            return ContentType.JSON_OBJECT
        except (TypeError, ValueError):
            pass

    if isinstance(value, types.ModuleType):
        return ContentType.MODULE_IMPORT

    if _is_cell_module_instance(value):
        return ContentType.MODULE_CELL_INSTANCE

    return ContentType.PICKLE_OBJECT


def _is_arrow_representable(value: Any) -> bool:
    """Return whether *value* should flow through the unified arrow/ipc codec.

    Covers everything we know how to encode as an Arrow Table with
    schema metadata: tables (pyarrow / pandas), n-d numpy arrays, numpy
    scalars, and typed Python primitives that have a native Arrow
    representation (datetime family, Decimal, bytes) plus two that
    don't but that we handle via metadata tags (UUID, complex).

    pyarrow is a guaranteed dep of notebook venvs; pandas and numpy
    are only installed when the user adds them (they're in the
    [notebook] extra on the strata side, but not baked into generated
    notebook pyprojects). Their imports are guarded so missing-package
    notebooks still classify correctly.
    """
    import datetime as _dt
    from decimal import Decimal
    from uuid import UUID

    import pyarrow as pa

    if isinstance(value, (pa.Table, pa.RecordBatch)):
        return True
    try:
        import pandas as pd

        if isinstance(value, (pd.DataFrame, pd.Series)):
            return True
    except ImportError:
        pass
    try:
        import numpy as np

        if isinstance(value, (np.ndarray, np.generic)):
            return True
    except ImportError:
        pass
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time, _dt.timedelta)):
        return True
    if isinstance(value, (Decimal, bytes, bytearray, UUID)):
        return True
    if isinstance(value, complex):
        return True
    return False


def _is_display_variable_name(variable_name: str | None) -> bool:
    """Return whether a variable name represents a display-only value."""
    return variable_name == "_" or (
        isinstance(variable_name, str) and variable_name.startswith("__display__")
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


_SerializeFn = Callable[[Any, Path, str], SerializedPayload]
_DeserializeFn = Callable[[Path], Any]


class _Handler(NamedTuple):
    """Bidirectional codec for one content type.

    ``serialize`` is ``None`` for content types that arrive on disk by
    other means (``module/cell`` is written by the module-export
    machinery, not by ``serialize_value``). ``deserialize`` is ``None``
    for display-only types (``image/png``) that the notebook UI
    consumes directly without round-tripping through Python.

    NamedTuple rather than ``@dataclass(frozen=True)`` because this
    module is loaded via ``importlib.util.spec_from_file_location`` in
    harness / pool_worker / inspect_repl subprocesses — that loader
    doesn't register the module in ``sys.modules`` before class
    creation runs, and ``dataclass`` crashes when it tries to look up
    annotations via ``sys.modules[cls.__module__]``. NamedTuple has
    no such introspection.
    """

    serialize: _SerializeFn | None
    deserialize: _DeserializeFn | None


def serialize_value(value: Any, output_dir: Path | str, variable_name: str) -> SerializedPayload:
    """Serialize *value* to *output_dir* and return a metadata dict.

    The metadata dict always contains:
      content_type  — one of the supported content types above
      file          — filename written (relative to output_dir)
      bytes         — file size in bytes
      preview       — a JSON-safe preview of the value
    Arrow results additionally include ``rows`` and ``columns``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    content_type = detect_content_type(value, variable_name)
    handler = _HANDLERS.get(content_type)
    if handler is None or handler.serialize is None:
        # detect_content_type only returns content types with a
        # registered serializer, so this is a defensive guard against
        # registry drift, not a path users can reach in practice.
        return _serialize_pickle(value, output_dir, variable_name)
    return handler.serialize(value, output_dir, variable_name)


def _serialize_arrow_with_fallback(
    value: Any, output_dir: Path, variable_name: str
) -> SerializedPayload:
    """Try the unified Arrow path; fall back to JSON-tagged-arrow or pickle on failure.

    Pandas-specific Arrow failures (Series shape mismatch, mixed dtypes
    pa.Table.from_pandas can't coerce) use the JSON table fallback so
    downstream code still sees a table-shaped artifact. Non-pandas
    failures (complex ndarray, structured dtype, unencodable scalar)
    pickle — the JSON fallback assumes a pandas shape and silently
    drops non-pandas values.
    """
    try:
        return _serialize_arrow(value, output_dir, variable_name)
    except Exception as exc:
        if _is_pandas_value(value) and _should_fallback_from_arrow_error(exc):
            logger.warning(
                "Arrow serialization of '%s' (%s) failed (%s); falling back "
                "to JSON-tagged-arrow. Direct-Arrow consumers (DuckDB/Polars) "
                "won't be able to read this artifact.",
                variable_name,
                type(value).__name__,
                exc,
            )
            return _serialize_dataframe_json(value, output_dir, variable_name)
        logger.warning(
            "Arrow serialization of '%s' (%s) failed (%s); falling back to "
            "pickle. Downstream cells expecting tabular shape will break.",
            variable_name,
            type(value).__name__,
            exc,
        )
        return _serialize_pickle(value, output_dir, variable_name)


# Schema-metadata keys for the unified arrow/ipc format. The shape key
# tells the reader which reconstruction path to take; the rest are
# shape-specific. Both keys and values are wire-stable — changing any
# byte invalidates already-cached arrow/ipc artifacts.
_META_SHAPE = b"strata.arrow.shape"
_META_SOURCE = b"strata.arrow.source"
_META_PD_NAME = b"strata.arrow.pandas.name"  # Series name
_META_TENSOR_SHAPE = b"strata.arrow.tensor.shape"  # JSON-encoded list[int]
_META_TENSOR_DTYPE = b"strata.arrow.tensor.dtype"  # e.g. b"int32", b"float64"
_META_SCALAR_TYPE = b"strata.arrow.scalar.type"

# Values of _META_SHAPE.
_SHAPE_TABLE = b"table"
_SHAPE_TENSOR = b"tensor"
_SHAPE_SCALAR = b"scalar"

# Values of _META_SOURCE — only set for pandas-origin tables.
_SOURCE_PANDAS_DATAFRAME = b"pandas.DataFrame"
_SOURCE_PANDAS_SERIES = b"pandas.Series"

# Values of _META_SCALAR_TYPE — set only for non-native typed scalars
# that need round-trip help beyond what pyarrow's native types give us.
_SCALAR_TYPE_UUID = b"uuid"
_SCALAR_TYPE_COMPLEX = b"complex"


def _serialize_arrow(value: Any, output_dir: Path, variable_name: str) -> SerializedPayload:
    """Unified writer for the arrow/ipc codec.

    Dispatches to one of three shape encoders (table / tensor / scalar)
    and stamps schema metadata so the reader can reconstruct the exact
    Python type. All three shapes use the same on-disk format — an
    Arrow IPC stream of a single Table — so readers only need one
    entry point.
    """
    import pyarrow as pa

    table = _to_arrow_table(value)

    filename = f"{variable_name}.arrow"
    filepath = output_dir / filename
    with open(filepath, "wb") as f:
        writer = pa.ipc.new_stream(f, table.schema)
        writer.write_table(table)
        writer.close()

    meta = table.schema.metadata or {}
    shape = meta.get(_META_SHAPE, _SHAPE_TABLE)

    if shape == _SHAPE_TENSOR:
        tensor_shape = json.loads(meta.get(_META_TENSOR_SHAPE, b"[]").decode("utf-8"))
        tensor_dtype = meta.get(_META_TENSOR_DTYPE, b"").decode("utf-8")
        preview = f"ndarray shape={tuple(tensor_shape)} dtype={tensor_dtype}"
        return {
            "content_type": ContentType.ARROW_IPC,
            "file": filename,
            "bytes": filepath.stat().st_size,
            "preview": preview,
        }

    if shape == _SHAPE_SCALAR:
        scalar_value = _extract_scalar_from_table(table)
        return {
            "content_type": ContentType.ARROW_IPC,
            "file": filename,
            "bytes": filepath.stat().st_size,
            "preview": to_serialization_safe(scalar_value),
        }

    # shape == table
    preview = []
    for i in range(min(20, table.num_rows)):
        preview.append([to_serialization_safe(col[i].as_py()) for col in table.columns])
    return {
        "content_type": ContentType.ARROW_IPC,
        "file": filename,
        "rows": table.num_rows,
        "columns": table.column_names,
        "bytes": filepath.stat().st_size,
        "preview": preview,
    }


def _to_arrow_table(value: Any) -> Any:
    """Dispatch *value* to an Arrow Table with shape metadata stamped.

    Tries each shape-specific converter in order; the first that
    accepts the value (returns non-None) wins. Each converter owns its
    own lazy import so pandas-free notebooks keep working.
    """
    for converter in _ARROW_CONVERTERS:
        table = converter(value)
        if table is not None:
            return table
    raise ValueError(f"Cannot convert {type(value).__name__} to Arrow")


def _arrow_from_pyarrow(value: Any) -> Any | None:
    import pyarrow as pa

    if isinstance(value, pa.RecordBatch):
        return _stamp_shape(pa.Table.from_batches([value]), _SHAPE_TABLE)
    if isinstance(value, pa.Table):
        return _stamp_shape(value, _SHAPE_TABLE)
    return None


def _arrow_from_pandas(value: Any) -> Any | None:
    try:
        import pandas as pd
    except ImportError:
        return None
    import pyarrow as pa

    if isinstance(value, pd.DataFrame):
        table = pa.Table.from_pandas(value)
        return _stamp_metadata(
            table, {_META_SHAPE: _SHAPE_TABLE, _META_SOURCE: _SOURCE_PANDAS_DATAFRAME}
        )
    if isinstance(value, pd.Series):
        # pa.Table.from_pandas expects a DataFrame — calling it with a
        # Series historically raised AttributeError. Promote to a
        # single-column frame and stash the original name so the
        # deserializer can round-trip back to Series.
        frame = value.to_frame()
        table = pa.Table.from_pandas(frame)
        return _stamp_metadata(
            table,
            {
                _META_SHAPE: _SHAPE_TABLE,
                _META_SOURCE: _SOURCE_PANDAS_SERIES,
                _META_PD_NAME: str(value.name or "").encode("utf-8"),
            },
        )
    return None


def _arrow_from_numpy(value: Any) -> Any | None:
    try:
        import numpy as np
    except ImportError:
        return None

    if isinstance(value, np.ndarray):
        return _ndarray_to_table(value)
    if isinstance(value, np.generic):
        # numpy scalar — lose the numpy flavor on round-trip, treat as
        # the equivalent Python primitive. Users who depend on
        # type(x) is np.int64 are vanishingly rare.
        return _python_scalar_to_table(value.item())
    return None


def _arrow_from_typed_scalar(value: Any) -> Any | None:
    """Wrap a typed Python primitive in a 1-row scalar Table.

    UUID and complex need a custom Arrow representation (binary(16) and
    a struct of floats) plus a scalar-type tag so the reader can
    reconstruct the original Python type. datetime / Decimal / bytes
    round-trip through pyarrow natively, so they just need the scalar
    shape tag and no type discriminator.
    """
    import datetime as _dt
    from decimal import Decimal
    from uuid import UUID

    import pyarrow as pa

    if isinstance(value, UUID):
        pa_arr = pa.array([value.bytes], type=pa.binary(16))
        table = pa.table({"value": pa_arr})
        return _stamp_metadata(
            table, {_META_SHAPE: _SHAPE_SCALAR, _META_SCALAR_TYPE: _SCALAR_TYPE_UUID}
        )

    if isinstance(value, complex):
        struct_type = pa.struct([("real", pa.float64()), ("imag", pa.float64())])
        pa_arr = pa.array([{"real": value.real, "imag": value.imag}], type=struct_type)
        table = pa.table({"value": pa_arr})
        return _stamp_metadata(
            table, {_META_SHAPE: _SHAPE_SCALAR, _META_SCALAR_TYPE: _SCALAR_TYPE_COMPLEX}
        )

    _dt_types = (_dt.datetime, _dt.date, _dt.time, _dt.timedelta)
    if isinstance(value, _dt_types) or isinstance(value, (Decimal, bytes, bytearray)):
        return _python_scalar_to_table(value)
    return None


# Order matters: pyarrow first (cheapest no-op), then pandas / numpy
# (their isinstance checks need lazy imports), then typed Python
# primitives.
_ARROW_CONVERTERS = (
    _arrow_from_pyarrow,
    _arrow_from_pandas,
    _arrow_from_numpy,
    _arrow_from_typed_scalar,
)


def _python_scalar_to_table(value: Any) -> Any:
    """Wrap a single primitive in a 1-row, 1-column Table via pa.array."""
    import pyarrow as pa

    pa_arr = pa.array([value])
    table = pa.table({"value": pa_arr})
    return _stamp_metadata(table, {_META_SHAPE: _SHAPE_SCALAR})


def _ndarray_to_table(arr: Any) -> Any:
    """Encode an ndarray as a 1-column Table + tensor shape metadata."""
    import numpy as np
    import pyarrow as pa

    contiguous = np.ascontiguousarray(arr)
    flat = contiguous.reshape(-1)
    pa_arr = pa.array(flat)
    table = pa.table({"values": pa_arr})
    return _stamp_metadata(
        table,
        {
            _META_SHAPE: _SHAPE_TENSOR,
            _META_TENSOR_SHAPE: json.dumps(list(contiguous.shape)).encode("utf-8"),
            _META_TENSOR_DTYPE: str(contiguous.dtype).encode("utf-8"),
        },
    )


def _stamp_shape(table: Any, shape: bytes) -> Any:
    return _stamp_metadata(table, {_META_SHAPE: shape})


def _stamp_metadata(table: Any, extra: dict[bytes, bytes]) -> Any:
    meta = dict(table.schema.metadata or {})
    meta.update(extra)
    return table.replace_schema_metadata(meta)


def _extract_scalar_from_table(table: Any) -> Any:
    """Read back a single Python scalar from a 1-row, 1-column Table."""
    meta = table.schema.metadata or {}
    scalar_type = meta.get(_META_SCALAR_TYPE, b"")
    col = table.column(0)
    raw = col[0].as_py()

    if scalar_type == _SCALAR_TYPE_UUID:
        from uuid import UUID

        return UUID(bytes=raw)
    if scalar_type == _SCALAR_TYPE_COMPLEX:
        return complex(raw["real"], raw["imag"])
    return raw


def _should_fallback_from_arrow_error(exc: Exception) -> bool:
    """Return whether Arrow serialization errors should use the JSON table fallback."""
    if isinstance(exc, (ImportError, ValueError, AttributeError)):
        return True

    try:
        import pyarrow as pa
    except ImportError:
        return False

    return isinstance(exc, pa.ArrowException)


def _is_pandas_value(value: Any) -> bool:
    """Return True when *value* is a pandas DataFrame or Series."""
    try:
        import pandas as pd
    except ImportError:
        return False
    return isinstance(value, (pd.DataFrame, pd.Series))


def to_serialization_safe(value: Any) -> Any:
    """Return a JSON- and TOML-compatible form of *value*.

    This is the **single sanitization boundary** for all downstream
    writers (manifest.json, notebook.toml, REST/WS response payloads).
    Callers can trust that the output contains only ``bool``, ``int``,
    ``float``, ``str``, ``list``, or ``dict[str, ...]`` — no ``None``,
    ``datetime``, ``Decimal``, ``bytes``, numpy scalars, or other types
    that trip up ``json.dump`` or ``tomli_w.dump``.

    Rules:
      - ``None`` → ``""`` (TOML rejects ``None``; empty string is safe
        for both JSON and TOML, preserves preview shape)
      - ``bool``, ``int``, ``float``, ``str`` → pass through
      - ``list`` / ``tuple`` → recursed list
      - ``dict`` → recursed dict with stringified keys
      - Everything else → ``str(value)``
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value
    # numpy 2.0+ scalars no longer subclass Python int/float.
    # Convert to native Python types before they reach JSON/TOML writers.
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
    except ImportError:
        pass
    if isinstance(value, (list, tuple)):
        return [to_serialization_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): to_serialization_safe(v) for k, v in value.items()}
    return str(value)


def _is_png_display_value(value: Any) -> bool:
    repr_png = getattr(value, "_repr_png_", None)
    if callable(repr_png):
        return True

    try:
        from matplotlib.figure import Figure

        if isinstance(value, Figure):
            return True
    except ImportError:
        pass

    try:
        from PIL import Image as _PILImage

        if isinstance(value, _PILImage.Image):
            return True
    except ImportError:
        pass

    return False


def _is_markdown_display_value(value: Any) -> bool:
    repr_markdown = getattr(value, "_repr_markdown_", None)
    return callable(repr_markdown)


def _coerce_markdown_text(value: Any) -> str:
    repr_markdown = getattr(value, "_repr_markdown_", None)
    if not callable(repr_markdown):
        raise ValueError(f"Cannot serialize {type(value)} as text/markdown")

    raw = repr_markdown()
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    if isinstance(raw, str):
        return raw
    raise ValueError("_repr_markdown_() must return str or UTF-8 bytes")


def _serialize_markdown(value: Any, output_dir: Path, variable_name: str) -> SerializedPayload:
    markdown_text = _coerce_markdown_text(value)
    filename = f"{variable_name}.md"
    filepath = output_dir / filename
    filepath.write_text(markdown_text, encoding="utf-8")
    return {
        "content_type": ContentType.TEXT_MARKDOWN,
        "file": filename,
        "bytes": filepath.stat().st_size,
        "markdown_text": markdown_text,
        "preview": None,
    }


def _serialize_image_png(value: Any, output_dir: Path, variable_name: str) -> SerializedPayload:
    for handler in _PNG_HANDLERS:
        result = handler(value)
        if result is not None:
            png_bytes, width, height = result
            break
    else:
        raise ValueError(f"Cannot serialize {type(value)} as image/png")

    if width is None or height is None:
        width, height = _png_size_from_bytes(png_bytes)

    filename = f"{variable_name}.png"
    filepath = output_dir / filename
    with open(filepath, "wb") as f:
        f.write(png_bytes)

    return {
        "content_type": ContentType.IMAGE_PNG,
        "file": filename,
        "bytes": filepath.stat().st_size,
        "inline_data_url": (f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}"),
        "width": width,
        "height": height,
        "preview": None,
    }


# Each handler returns ``(png_bytes, width|None, height|None)`` on a
# match or ``None`` to defer to the next handler. Order is by cost +
# specificity: caller-provided ``_repr_png_`` first, then matplotlib
# (heaviest typical case), then PIL.
_PngHandlerResult = tuple[bytes, int | None, int | None]


def _png_via_repr_png(value: Any) -> _PngHandlerResult | None:
    repr_png = getattr(value, "_repr_png_", None)
    if not callable(repr_png):
        return None
    raw = repr_png()
    if isinstance(raw, str):
        return raw.encode("latin1"), None, None
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return bytes(raw), None, None
    if raw is None:
        return None
    raise ValueError("_repr_png_() must return bytes-like data")


def _png_via_matplotlib(value: Any) -> _PngHandlerResult | None:
    try:
        from matplotlib.figure import Figure
    except ImportError:
        return None
    if not isinstance(value, Figure):
        return None
    buffer = io.BytesIO()
    value.savefig(buffer, format="png")
    width = int(round(value.get_figwidth() * value.dpi))
    height = int(round(value.get_figheight() * value.dpi))
    return buffer.getvalue(), width, height


def _png_via_pil(value: Any) -> _PngHandlerResult | None:
    try:
        from PIL import Image as _PILImage
    except ImportError:
        return None
    if not isinstance(value, _PILImage.Image):
        return None
    buffer = io.BytesIO()
    value.save(buffer, format="PNG")
    width, height = value.size
    return buffer.getvalue(), width, height


_PNG_HANDLERS = (_png_via_repr_png, _png_via_matplotlib, _png_via_pil)


def _png_size_from_bytes(png_bytes: bytes) -> tuple[int | None, int | None]:
    """Probe PIL for size when the source handler couldn't supply one.

    Returns ``(None, None)`` if PIL isn't installed or fails to read
    the bytes; the caller stamps the dimensions as ``None`` in that
    case so downstream renderers fall back to intrinsic sizing.
    """
    try:
        from PIL import Image as _PILImage
    except ImportError:
        return None, None
    try:
        with _PILImage.open(io.BytesIO(png_bytes)) as image:
            return image.size
    except Exception:
        return None, None


def _serialize_dataframe_json(
    value: Any, output_dir: Path, variable_name: str
) -> SerializedPayload:
    """JSON fallback for DataFrames when Arrow serialization fails.

    The fallback still uses ``arrow/ipc`` metadata and a ``.arrow`` artifact
    name so downstream dependency loading treats the value as table-shaped.
    The file contents are JSON, tagged with a serializer-local marker that
    ``_deserialize_arrow`` understands even when ``pyarrow`` is unavailable.
    """
    payload: dict[str, Any] = {
        _TAG_ARROW_JSON_FALLBACK: True,
        "format": _ARROW_JSON_FALLBACK_TAG,
        "kind": "dataframe",
        "columns": [],
        "data": [],
        "series_name": None,
    }

    try:
        import pandas as pd

        is_series = isinstance(value, pd.Series)
        frame = value.to_frame() if is_series else value
        columns = [to_serialization_safe(column) for column in list(frame.columns)]
        num_rows = len(frame)
        rows = [
            [to_serialization_safe(v) for v in row]
            for row in frame.itertuples(index=False, name=None)
        ]
        preview = rows[:20]
        payload.update(
            {
                "kind": "series" if is_series else "dataframe",
                "columns": columns,
                "data": rows,
                "series_name": to_serialization_safe(value.name) if is_series else None,
            }
        )
    except Exception:
        columns = []
        num_rows = 0
        preview = []
        payload.update({"columns": [], "data": [], "series_name": None})

    filename = f"{variable_name}.arrow"
    filepath = output_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    return {
        "content_type": ContentType.ARROW_IPC,
        "file": filename,
        "rows": num_rows,
        "columns": columns,
        "bytes": filepath.stat().st_size,
        "preview": preview,
    }


def _serialize_json(value: Any, output_dir: Path, variable_name: str) -> SerializedPayload:
    filename = f"{variable_name}.json"
    filepath = output_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
    return {
        "content_type": ContentType.JSON_OBJECT,
        "file": filename,
        "bytes": filepath.stat().st_size,
        "preview": value,
    }


def _serialize_module(value: Any, output_dir: Path, variable_name: str) -> SerializedPayload:
    module_name = getattr(value, "__name__", variable_name)
    filename = f"{variable_name}.module.json"
    filepath = output_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"module_name": module_name}, f)
    return {
        "content_type": ContentType.MODULE_IMPORT,
        "file": filename,
        "bytes": filepath.stat().st_size,
        "preview": f"<module '{module_name}'>",
    }


def _serialize_cell_instance(value: Any, output_dir: Path, variable_name: str) -> SerializedPayload:
    module = sys.modules.get(type(value).__module__)
    module_source = getattr(module, _CELL_MODULE_SOURCE_ATTR, None)
    if not isinstance(module_source, str) or not module_source:
        raise ValueError(
            f"Cannot serialize notebook-exported instance '{variable_name}' "
            "because its synthetic module source is unavailable"
        )

    state = _extract_cell_instance_state(value)
    codec = _resolve_object_codec()
    state_bytes = codec.dumps(state)
    filename = f"{variable_name}.cell_instance.pickle"
    filepath = output_dir / filename
    payload = {
        "module_name": type(value).__module__,
        "class_name": type(value).__name__,
        "source": module_source,
        "state_codec": codec.name,
        "state_payload": state_bytes,
    }
    with open(filepath, "wb") as f:
        pickle.dump(payload, f, protocol=5)

    type_name = type(value).__name__
    return {
        "content_type": ContentType.MODULE_CELL_INSTANCE,
        "file": filename,
        "bytes": filepath.stat().st_size,
        "codec": codec.name,
        "type": type_name,
        "preview": f"<{type_name} object>",
    }


def _serialize_pickle(value: Any, output_dir: Path, variable_name: str) -> SerializedPayload:
    # Pickle failures used to return a success-shape dict with file=None,
    # which left the artifact store believing the value had been written
    # while the next reader hit a missing file. Raise instead — both
    # harness.py and pool_worker.py wrap serialize_value() in a
    # try/except that converts the exception into an error-shape entry,
    # so the failure surfaces at the cell that produced it.
    filename = f"{variable_name}.pickle"
    filepath = output_dir / filename
    codec = _resolve_object_codec()
    payload = codec.dumps(value)
    envelope = _wrap_codec_payload(codec.name, payload)
    with open(filepath, "wb") as f:
        pickle.dump(envelope, f, protocol=5)
    return {
        "content_type": ContentType.PICKLE_OBJECT,
        "file": filename,
        "bytes": filepath.stat().st_size,
        "codec": codec.name,
        "type": type(value).__name__,
        "preview": f"<{type(value).__name__} object>",
    }


# ---------------------------------------------------------------------------
# Deserialization
# ---------------------------------------------------------------------------

# Extension → content-type mapping (also used by executor._store_outputs)
EXT_TO_CONTENT_TYPE: dict[str, ContentType] = {
    ".arrow": ContentType.ARROW_IPC,
    ".md": ContentType.TEXT_MARKDOWN,
    ".json": ContentType.JSON_OBJECT,
    ".pickle": ContentType.PICKLE_OBJECT,
    ".module.json": ContentType.MODULE_IMPORT,
    ".cell_module.json": ContentType.MODULE_CELL,
    ".cell_instance.pickle": ContentType.MODULE_CELL_INSTANCE,
    ".rds": ContentType.RDS_OBJECT,
}


def deserialize_value(
    content_type: str, file_path: Path | str, output_dir: Path | str | None = None
) -> Any:
    """Deserialize a value from *file_path*.

    *output_dir* is accepted for API compatibility but not required —
    *file_path* is always treated as an absolute (or relative-to-cwd) path.
    Accepts either a ContentType enum value or the raw string form; since
    ContentType is a StrEnum, ``_HANDLERS`` lookups work for both.
    """
    handler = _HANDLERS.get(content_type)
    if handler is None or handler.deserialize is None:
        raise ValueError(f"Unknown content type: {content_type!r}")
    return handler.deserialize(Path(file_path))


def _deserialize_arrow(file_path: Path) -> Any:
    """Read an Arrow IPC stream.

    Branches on the ``strata.arrow.shape`` schema-metadata tag to
    reconstruct the original Python type: table (DataFrame/Series/
    pa.Table), tensor (numpy ndarray with original shape+dtype), or
    scalar (typed primitive).
    """
    fallback_payload = _read_arrow_json_fallback(file_path)
    if fallback_payload is not None:
        return _deserialize_arrow_json_fallback(fallback_payload)

    import pyarrow as pa

    with open(file_path, "rb") as f:
        reader = pa.ipc.open_stream(f)
        table = reader.read_all()

    # Consolidate chunks so downstream code that creates RecordBatches
    # (e.g. DataFusion's register_record_batches) gets flat Arrays
    # instead of ChunkedArrays. Without this, pa.RecordBatch.from_pandas
    # fails on DataFrames that survived an Arrow round-trip.
    table = table.combine_chunks()

    meta = table.schema.metadata or {}
    shape = meta.get(_META_SHAPE, _SHAPE_TABLE)

    if shape == _SHAPE_TENSOR:
        return _tensor_from_table(table)

    if shape == _SHAPE_SCALAR:
        return _extract_scalar_from_table(table)

    # Default / table shape — pandas DataFrame or pa.Table.
    return _table_to_pandas_or_arrow(table)


def _table_to_pandas_or_arrow(table: Any) -> Any:
    """Decode a shape=table Arrow Table back to pandas or pyarrow.

    When the source metadata says the value originated as pandas but
    to_pandas() fails, we still return the pa.Table — but log it. Silent
    type changes break downstream cells that called ``.iloc`` on what
    they expected to be a DataFrame, and the AttributeError they see
    gives no clue why the type morphed across the round-trip.
    """
    meta = table.schema.metadata or {}
    source = meta.get(_META_SOURCE, b"")

    try:
        frame = table.to_pandas()
    except Exception as exc:
        if source in (_SOURCE_PANDAS_DATAFRAME, _SOURCE_PANDAS_SERIES):
            logger.warning(
                "Arrow→pandas conversion failed (%s); returning pa.Table even "
                "though the value originated as %s. Downstream cells expecting "
                "pandas methods will fail with AttributeError.",
                exc,
                source.decode("utf-8"),
            )
        return table

    if source == _SOURCE_PANDAS_SERIES:
        try:
            series = frame.iloc[:, 0]
            name_bytes = meta.get(_META_PD_NAME, b"")
            series.name = name_bytes.decode("utf-8") or None
            return series
        except Exception as exc:
            logger.warning(
                "Reconstructing pandas.Series from Arrow failed (%s); returning DataFrame instead.",
                exc,
            )
            return frame
    return frame


def _tensor_from_table(table: Any) -> Any:
    """Decode a shape=tensor Arrow Table back to a numpy ndarray."""
    import numpy as np

    meta = table.schema.metadata or {}
    raw_shape = meta.get(_META_TENSOR_SHAPE, b"[]").decode("utf-8")
    dtype_str = meta.get(_META_TENSOR_DTYPE, b"").decode("utf-8")
    shape = tuple(json.loads(raw_shape))

    flat = table.column(0).to_numpy(zero_copy_only=False)
    if dtype_str:
        flat = flat.astype(np.dtype(dtype_str), copy=False)
    return np.ascontiguousarray(flat).reshape(shape)


def _read_arrow_json_fallback(file_path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get(_TAG_ARROW_JSON_FALLBACK) is not True:
        return None
    if payload.get("format") != _ARROW_JSON_FALLBACK_TAG:
        return None
    return payload


def _deserialize_arrow_json_fallback(payload: dict[str, Any]) -> Any:
    columns = payload.get("columns")
    rows = payload.get("data")
    kind = payload.get("kind")
    series_name = payload.get("series_name")

    if not isinstance(columns, list) or not isinstance(rows, list):
        raise ValueError("Invalid notebook JSON table fallback payload")

    try:
        import pandas as pd
    except ImportError:
        return payload

    frame = pd.DataFrame(rows, columns=columns)
    if kind == "series":
        if frame.shape[1] == 0:
            series = pd.Series(dtype=object)
        else:
            series = frame.iloc[:, 0]
        series.name = series_name
        return series
    return frame


def _deserialize_json(file_path: Path) -> Any:
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def _deserialize_markdown(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def _deserialize_rds(file_path: Path) -> Any:
    # RDS payloads are R's native binary format; Python has no reader.
    # Raise the structured error here so any Python caller that tries
    # to consume an R-only artifact gets the suggested-fix message
    # instead of a generic "Unknown content type" or bytes garbage.
    raise StrataRArtifactError(file_path)


def _deserialize_pickle(file_path: Path) -> Any:
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    codec_payload = _unwrap_codec_payload(data)
    if codec_payload is None:
        # Backward compatibility: historical notebook artifacts stored raw pickle payloads.
        return data

    codec_name, payload = codec_payload
    codec = _resolve_object_codec(codec_name)
    return codec.loads(payload)


def _deserialize_module(file_path: Path) -> Any:
    import importlib

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    return importlib.import_module(data["module_name"])


def _ensure_cell_module(
    module_name: str,
    module_source: str,
    file_path: Path,
):
    import types

    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        module.__file__ = str(file_path)
        sys.modules[module_name] = module
        exec(compile(module_source, module_name, "exec"), module.__dict__)  # noqa: S102
    module.__dict__[_CELL_MODULE_SOURCE_ATTR] = module_source
    module.__dict__[_CELL_MODULE_FLAG_ATTR] = True
    for value in module.__dict__.values():
        if isinstance(value, type) and getattr(value, "__module__", None) == module_name:
            try:
                setattr(value, _CELL_EXPORTED_CLASS_ATTR, True)
            except (AttributeError, TypeError):
                continue
    return module


def _deserialize_cell_module(file_path: Path) -> Any:
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    module_name = data.get("module_name")
    symbol_name = data.get("symbol_name")
    module_source = data.get("source")
    if not isinstance(module_name, str) or not isinstance(symbol_name, str):
        raise ValueError("Invalid exported notebook module descriptor")
    if not isinstance(module_source, str):
        raise ValueError(f"Exported notebook module '{module_name}' has invalid source")

    module = _ensure_cell_module(module_name, module_source, file_path)

    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ValueError(
            f"Exported notebook module '{module_name}' does not define '{symbol_name}'"
        ) from exc


def _deserialize_cell_instance(file_path: Path) -> Any:
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError("Invalid notebook-exported instance payload")

    module_name = data.get("module_name")
    class_name = data.get("class_name")
    module_source = data.get("source")
    if not isinstance(module_name, str) or not isinstance(class_name, str):
        raise ValueError("Invalid notebook-exported instance descriptor")
    if not isinstance(module_source, str):
        raise ValueError(f"Exported notebook instance '{class_name}' has invalid module source")

    module = _ensure_cell_module(module_name, module_source, file_path)
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(
            f"Exported notebook module '{module_name}' does not define class '{class_name}'"
        ) from exc

    if "state_payload" in data and "state_codec" in data:
        state_codec = data["state_codec"]
        state_payload = data["state_payload"]
        if not isinstance(state_codec, str) or not isinstance(state_payload, bytes):
            raise ValueError("Invalid notebook-exported instance state payload")
        state = _resolve_object_codec(state_codec).loads(state_payload)
    else:
        # Backward compatibility for the first module/cell-instance format.
        state_pickle = data["state_pickle"]
        state = pickle.loads(state_pickle)
    instance = cls.__new__(cls)

    setstate = getattr(instance, "__setstate__", None)
    if callable(setstate):
        setstate(state)
    elif _is_default_cell_instance_state(state):
        _restore_default_cell_instance_state(instance, state)
    elif state is None:
        pass
    elif isinstance(state, dict):
        instance.__dict__.update(state)
    else:
        raise ValueError(
            f"Cannot restore notebook-exported instance '{class_name}' without __setstate__"
        )

    return instance


def _is_cell_module_instance(value: Any) -> bool:
    if isinstance(value, type):
        return False

    module = sys.modules.get(type(value).__module__)
    module_source = getattr(module, _CELL_MODULE_SOURCE_ATTR, None)
    return bool(
        getattr(type(value), _CELL_EXPORTED_CLASS_ATTR, False)
        and isinstance(module_source, str)
        and module_source
    )


def _extract_cell_instance_state(value: Any) -> Any:
    getstate = getattr(type(value), "__getstate__", None)
    if callable(getstate) and getstate is not object.__getstate__:
        return value.__getstate__()

    return _extract_default_cell_instance_state(value)


def _extract_default_cell_instance_state(value: Any) -> Any:
    dict_state = dict(value.__dict__) if hasattr(value, "__dict__") else None
    slot_state: dict[str, Any] = {}
    for slot_name in _iter_slot_names(type(value)):
        try:
            slot_state[slot_name] = getattr(value, slot_name)
        except AttributeError:
            continue

    if dict_state is None and not slot_state:
        return None

    return {
        _TAG_CELL_INSTANCE_STATE: _CELL_INSTANCE_STATE_TAG,
        "dict": dict_state,
        "slots": slot_state,
    }


def _is_default_cell_instance_state(state: Any) -> bool:
    return (
        isinstance(state, dict) and state.get(_TAG_CELL_INSTANCE_STATE) == _CELL_INSTANCE_STATE_TAG
    )


def _restore_default_cell_instance_state(instance: Any, state: Any) -> None:
    if not isinstance(state, dict):
        raise ValueError("Invalid notebook-exported instance state")

    dict_state = state.get("dict")
    slot_state = state.get("slots")

    if dict_state is not None:
        if not isinstance(dict_state, dict):
            raise ValueError("Invalid notebook-exported instance __dict__ state")
        instance.__dict__.update(dict_state)

    if slot_state is not None:
        if not isinstance(slot_state, dict):
            raise ValueError("Invalid notebook-exported instance __slots__ state")
        for slot_name, slot_value in slot_state.items():
            if not isinstance(slot_name, str):
                raise ValueError("Invalid notebook-exported instance slot name")
            setattr(instance, slot_name, slot_value)


def _iter_slot_names(cls: type[Any]) -> list[str]:
    slot_names: list[str] = []
    for klass in cls.__mro__:
        slots = klass.__dict__.get("__slots__")
        if slots is None:
            continue
        if isinstance(slots, str):
            slot_values = [slots]
        else:
            slot_values = list(slots)
        for slot_name in slot_values:
            if slot_name in {"__dict__", "__weakref__"}:
                continue
            if slot_name not in slot_names:
                slot_names.append(slot_name)

    return slot_names


# ---------------------------------------------------------------------------
# Handler registry — the canonical map from ContentType to (serialize,
# deserialize). Lives at module bottom so every handler function is
# defined before being referenced. Keyed by ``str`` so both the
# ``ContentType`` enum and raw string content-types (the form callers
# from outside this module typically pass) resolve to the same entry.
# ---------------------------------------------------------------------------

_HANDLERS: dict[str, _Handler] = {
    ContentType.ARROW_IPC: _Handler(_serialize_arrow_with_fallback, _deserialize_arrow),
    ContentType.JSON_OBJECT: _Handler(_serialize_json, _deserialize_json),
    ContentType.PICKLE_OBJECT: _Handler(_serialize_pickle, _deserialize_pickle),
    ContentType.IMAGE_PNG: _Handler(_serialize_image_png, None),
    ContentType.TEXT_MARKDOWN: _Handler(_serialize_markdown, _deserialize_markdown),
    ContentType.MODULE_IMPORT: _Handler(_serialize_module, _deserialize_module),
    ContentType.MODULE_CELL: _Handler(None, _deserialize_cell_module),
    ContentType.MODULE_CELL_INSTANCE: _Handler(
        _serialize_cell_instance, _deserialize_cell_instance
    ),
    # No serialize side — Python never produces RDS; the file is
    # always written by harness.R. The deserializer raises rather
    # than returning a value.
    ContentType.RDS_OBJECT: _Handler(None, _deserialize_rds),
}
