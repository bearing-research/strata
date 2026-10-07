"""Shared serialization and deserialization for notebook cell values.

Content types: ``arrow/ipc`` (anything Arrow-representable, plus types that
export ``__arrow_c_stream__`` or ``__dlpack__``; shape in the schema metadata
``strata.arrow.shape``), ``json/object``, ``image/png``, ``text/markdown``,
``module/import``, ``module/cell``, ``module/cell-instance``, ``pickle/object``
(everything else) and ``application/x-r-rds`` (R-only; Python refuses to read it).

harness.py, pool_worker.py, inspect_harness.py and cell_test_conftest.py load this
file by path with ``importlib.util``: they run in the notebook's venv and cannot
``import strata``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import pickle
import sys
from collections.abc import Callable, Iterable
from enum import StrEnum
from pathlib import Path
from typing import Any, NamedTuple, NotRequired, Protocol, TypedDict, cast

logger = logging.getLogger(__name__)


class ContentType(StrEnum):
    """Content type strings used by the notebook serializer.

    A StrEnum, so members compare equal to the plain strings. Defined here because
    this file is loaded where ``strata`` is not importable.
    """

    ARROW_IPC = "arrow/ipc"
    JSON_OBJECT = "json/object"
    PICKLE_OBJECT = "pickle/object"
    IMAGE_PNG = "image/png"
    TEXT_MARKDOWN = "text/markdown"
    MODULE_IMPORT = "module/import"
    MODULE_CELL = "module/cell"
    MODULE_CELL_INSTANCE = "module/cell-instance"
    # saveRDS() blob from harness.R for values outside the Arrow and JSON tiers.
    # Unreadable from Python; ``_deserialize_rds`` raises ``StrataRArtifactError``.
    RDS_OBJECT = "application/x-r-rds"
    # An ``@fetch`` delivered to a remote worker: written into the run directory,
    # and the cell gets the ``Path``, as locally.
    FILE_PATH = "file/path"


class StrataRArtifactError(RuntimeError):
    """Raised when a Python cell tries to consume an R-only RDS artifact.

    ``code`` is ``R_ONLY_ARTIFACT``; ``file_path`` is the RDS blob (debugging only);
    ``variable_name`` is the upstream variable, set by the harness when known.
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


_x64_enabled_here = False

# Kept beside the flag so the harness and pooled worker share one wording.
X64_NOTE = (
    "strata: turned on jax_enable_x64 to hand back a stored 64-bit array at its own "
    "width. That switch is process-wide, so in a reused worker every later cell gets "
    "64-bit defaults too, and no cell's provenance records which it got. Put "
    "`# @env JAX_ENABLE_X64=1` on the cells that want it: jax reads it at import, and "
    "it becomes part of their env hash.\n"
)


def _record_x64_enabled() -> None:
    """Remember that this process turned ``jax_enable_x64`` on, once."""
    global _x64_enabled_here
    _x64_enabled_here = True


def x64_was_enabled_here() -> bool:
    """Whether reconstructing an input turned ``jax_enable_x64`` on.

    The switch is process-wide, so a reused worker (warm pool, Run All batch) keeps
    it on for later cells, which then get float64 instead of float32 without their
    provenance recording it. Setting ``JAX_ENABLE_X64`` in the notebook env is the
    deliberate way (it is in the env hash); the caller reports this to suggest it.
    """
    return _x64_enabled_here


class StrataPrecisionError(RuntimeError):
    """Raised when a stored array cannot be reconstructed at its own dtype.

    A backstop: the reader enables ``jax_enable_x64`` and converts again when a
    reconstruction comes back narrowed, and raises only if the dtype is one this
    JAX cannot represent. Silently handing back 32-bit values for 64-bit ones fails
    far away, e.g. as a ``lax.while_loop`` carry dtype mismatch.

    ``code`` is ``PRECISION_NARROWED``; ``stored_dtype`` and ``reconstructed`` are
    the recorded and produced dtypes; ``variable_name`` is set during input
    deserialization when known.
    """

    code = "PRECISION_NARROWED"

    def __init__(
        self,
        stored_dtype: str,
        reconstructed_dtype: str,
        *,
        variable_name: str | None = None,
    ) -> None:
        self.stored_dtype = stored_dtype
        self.reconstructed_dtype = reconstructed_dtype
        self.variable_name = variable_name
        target = f"variable '{variable_name}'" if variable_name else "a stored array"
        super().__init__(
            f"Reading {target} as a JAX array narrows {stored_dtype} to "
            f"{reconstructed_dtype}, even with jax_enable_x64 on, so the value "
            f"the upstream cell stored cannot be handed over unchanged. This "
            f"JAX build appears unable to represent {stored_dtype}."
        )


class SerializedPayload(TypedDict):
    """Metadata dict returned by ``serialize_value`` and every ``_serialize_*`` helper.

    ``content_type``, ``file``, ``bytes`` and ``preview`` are always present; the
    rest are content-type-specific extras.
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

# Wire-stable: changing any value invalidates every cached pickle,
# JSON-fallback and instance-state artifact.
_TAG_OBJECT_CODEC = "__strata_object_codec__"
_TAG_ARROW_JSON_FALLBACK = "__strata_arrow_json_fallback__"
_TAG_CELL_INSTANCE_STATE = "__strata_cell_instance_state__"

# Mark a synthetic cell-exported module and its exported classes.
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

    Default is cloudpickle (handles lambdas, closures, dynamic classes); setting
    the env var to ``pickle`` opts out. Falls back to stdlib pickle when cloudpickle
    is unavailable.
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


# --- Content-type detection ---


def _survives_json(value: Any) -> bool:
    """Whether JSON encoding would return *value* unchanged.

    ``json.dumps`` coerces rather than fails on non-string dict keys (``{1: "a"}``
    decodes as ``{"1": "a"}``) and tuples (decoded as lists). Called only after
    ``json.dumps`` succeeded, so the walk terminates. NaN and Inf are not losses:
    the JSON writer round-trips both.
    """
    if isinstance(value, dict):
        return all(isinstance(key, str) for key in value) and all(
            _survives_json(item) for item in value.values()
        )
    if isinstance(value, tuple):
        return False
    if isinstance(value, list):
        return all(_survives_json(item) for item in value)
    return True


def detect_content_type(value: Any, variable_name: str | None = None) -> ContentType:
    """Return the content type for *value*; the first match wins.

    1. Arrow-representable: ``arrow/ipc``
    2. Markdown / PNG display value: ``text/markdown`` or ``image/png``
    3. JSON-serializable primitive: ``json/object``
    4. Python module: ``module/import``
    5. Cell-defined class instance: ``module/cell-instance``
    6. Anything else: ``pickle/object``

    Imports stay lazy so loading this module does not pay pyarrow/pandas/numpy init.
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
        # Structurally JSON-safe but may hold NaN, Inf, non-string keys or nested
        # non-primitives. The probe write catches what the writer rejects;
        # ``_survives_json`` catches what it silently coerces.
        try:
            json.dumps(value)
            if _survives_json(value):
                return ContentType.JSON_OBJECT
        except (TypeError, ValueError):
            pass

    if isinstance(value, types.ModuleType):
        return ContentType.MODULE_IMPORT

    if _is_cell_module_instance(value):
        return ContentType.MODULE_CELL_INSTANCE

    return ContentType.PICKLE_OBJECT


def _is_arrow_representable(value: Any) -> bool:
    """Return whether *value* should flow through the arrow/ipc codec.

    True iff some :data:`_ARROW_TYPE_RULES` entry matches, the same registry
    :func:`_to_arrow_table` encodes with. The two generic rules at the end
    (``__arrow_c_stream__``, ``__dlpack__``) keep unrecognised libraries' values
    readable as Arrow; they come back as a pa.Table / ndarray, not the original type.
    """
    return any(rule.matches(value) for rule in _ARROW_TYPE_RULES)


def _matched_only_generic_rules(value: Any) -> bool:
    """Whether *value* reached the Arrow path on a protocol probe alone.

    Tells a genuine encoding failure (a DataFrame that could not become Arrow) from
    a protocol exporter we cannot convert, which keeps its pickle.
    """
    return not any(rule.matches(value) for rule in _NAMED_ARROW_TYPE_RULES)


def _is_display_variable_name(variable_name: str | None) -> bool:
    """Return whether a variable name represents a display-only value."""
    return variable_name == "_" or (
        isinstance(variable_name, str) and variable_name.startswith("__display__")
    )


# --- Serialization ---


_SerializeFn = Callable[[Any, Path, str], SerializedPayload]
_DeserializeFn = Callable[[Path], Any]


class _Handler(NamedTuple):
    """Bidirectional codec for one content type.

    ``serialize`` is ``None`` for types written by other means (``module/cell``);
    ``deserialize`` is ``None`` for display-only types (``image/png``). A NamedTuple
    because ``dataclass`` fails under the ``spec_from_file_location`` loader, which
    does not register the module in ``sys.modules`` before class creation.
    """

    serialize: _SerializeFn | None
    deserialize: _DeserializeFn | None


def _safe_filename_stem(variable_name: str) -> str:
    """Case-collision-proof filename stem (write side).

    A copy of ``provenance.safe_filename_stem`` (a test keeps them identical),
    because this module cannot ``import strata``.
    """
    if variable_name != variable_name.lower():
        return f"{variable_name}-{hashlib.sha256(variable_name.encode()).hexdigest()[:8]}"
    return variable_name


def serialize_value(value: Any, output_dir: Path | str, variable_name: str) -> SerializedPayload:
    """Serialize *value* to *output_dir* and return its metadata.

    The metadata always has ``content_type``, ``file`` (relative to *output_dir*),
    ``bytes`` and a JSON-safe ``preview``; Arrow results add ``rows`` and ``columns``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    content_type = detect_content_type(value, variable_name)
    handler = _HANDLERS.get(content_type)
    if handler is None or handler.serialize is None:
        # Defensive guard against registry drift; unreachable in practice.
        return _serialize_pickle(value, output_dir, variable_name)
    return handler.serialize(value, output_dir, variable_name)


def serialize_display_value(
    value: Any,
    output_dir: Path | str,
    index: int,
    written: Iterable[tuple[Any, SerializedPayload]] = (),
) -> SerializedPayload:
    """Serialize one display value, reusing a variable's payload for the same object.

    Writing it twice is wrong for anything whose bytes are not a pure function of
    the object: a lazy query re-runs, and a one-shot stream is already drained.
    *written* holds the ``(value, payload)`` pairs serialized so far. Reuse also
    needs the content type to agree: a figure can be ``pickle/object`` as a variable
    and ``image/png`` as a display, and is then written twice.
    """
    variable_name = f"__display__{index}"
    content_type = detect_content_type(value, variable_name)
    for written_value, payload in written:
        if (
            written_value is value
            and payload.get("content_type") == content_type
            and payload.get("file")
        ):
            # Display payloads are read by ``file`` and stored under a display-specific
            # id, so sharing the variable's file is safe.
            return cast(SerializedPayload, dict(payload))
    return serialize_value(value, output_dir, variable_name)


def _serialize_arrow_with_fallback(
    value: Any, output_dir: Path, variable_name: str
) -> SerializedPayload:
    """Try the unified Arrow path; fall back to JSON-tagged Arrow or pickle on failure.

    Pandas Arrow failures use the JSON table fallback so downstream still sees a
    table. Other failures pickle, because the JSON fallback assumes a pandas shape.
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
        if _matched_only_generic_rules(value):
            # Reached Arrow on a protocol probe alone (a non-struct pa.ChunkedArray, a
            # device-resident dlpack buffer). Pickle is correct here, not a degradation.
            logger.debug(
                "'%s' (%s) advertises an Arrow protocol but could not be "
                "converted (%s); storing it as pickle.",
                variable_name,
                type(value).__name__,
                exc,
            )
        else:
            logger.warning(
                "Arrow serialization of '%s' (%s) failed (%s); falling back to "
                "pickle. Downstream cells expecting tabular shape will break.",
                variable_name,
                type(value).__name__,
                exc,
            )
        return _serialize_pickle(value, output_dir, variable_name)


# The shape key picks the reader's reconstruction path. Keys and values are
# wire-stable: changing any byte invalidates cached arrow/ipc artifacts.
_META_SHAPE = b"strata.arrow.shape"
_META_SOURCE = b"strata.arrow.source"
_META_PD_NAME = b"strata.arrow.pandas.name"  # Series name


def _encode_series_name(name: Any) -> bytes:
    """JSON-encode a Series name, preserving ``None`` vs ``""`` vs ``0``.

    Non-JSON-safe names degrade to their string form.
    """
    if name is None or isinstance(name, (str, int, float, bool)):
        return json.dumps(name).encode("utf-8")
    return json.dumps(str(name)).encode("utf-8")


def _decode_series_name(raw: bytes) -> Any:
    """Decode a stored Series name; legacy blobs hold a plain string (``""`` = None)."""
    text = raw.decode("utf-8")
    try:
        return json.loads(text)
    except ValueError:
        return text or None


_META_TENSOR_SHAPE = b"strata.arrow.tensor.shape"  # JSON-encoded list[int]
_META_TENSOR_DTYPE = b"strata.arrow.tensor.dtype"  # e.g. b"int32", b"float64"
_META_SCALAR_TYPE = b"strata.arrow.scalar.type"

# Values of _META_SHAPE.
_SHAPE_TABLE = b"table"
_SHAPE_TENSOR = b"tensor"
_SHAPE_SCALAR = b"scalar"

# Values of _META_SOURCE: the originating library, so the reader rebuilds the
# exact type (degrading when it's absent). Table shape: pandas / polars.
# Tensor shape: torch / jax.
_SOURCE_PYARROW_TABLE = b"pyarrow.Table"
_SOURCE_PYARROW_RECORD_BATCH = b"pyarrow.RecordBatch"
_SOURCE_PANDAS_DATAFRAME = b"pandas.DataFrame"
_SOURCE_PANDAS_SERIES = b"pandas.Series"
_SOURCE_POLARS_DATAFRAME = b"polars.DataFrame"
_SOURCE_POLARS_SERIES = b"polars.Series"
_SOURCE_TORCH = b"torch.Tensor"
_SOURCE_JAX = b"jax.Array"
# Arrived via the Arrow PyCapsule interface: tabular, type unknown. A separate
# tag so lineage doesn't claim a pyarrow.Table origin.
_SOURCE_ARROW_CAPSULE = b"arrow.capsule"

# Values of _META_SCALAR_TYPE, set only for typed scalars pyarrow can't round-trip natively.
_SCALAR_TYPE_UUID = b"uuid"
_SCALAR_TYPE_COMPLEX = b"complex"

# Complex dtypes carried as interleaved real/imag, and the real view dtype.
# Only what JAX and torch produce; wider ones (clongdouble) stay pickled.
_REAL_VIEW_OF_COMPLEX = {"complex64": "float32", "complex128": "float64"}


def _serialize_arrow(value: Any, output_dir: Path, variable_name: str) -> SerializedPayload:
    """Unified writer for the arrow/ipc codec.

    Every shape (table / tensor / scalar) is written as an Arrow IPC stream of one
    Table, with schema metadata the reader uses to rebuild the exact Python type.
    """
    import pyarrow as pa

    table = _to_arrow_table(value)

    filename = f"{_safe_filename_stem(variable_name)}.arrow"
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


class _ArrowRule(NamedTuple):
    """One entry in the arrow/ipc type registry.

    ``matches`` is a cheap, side-effect-free predicate used both for detection and
    to pick the converter; ``to_table`` may assume it returned True. A NamedTuple for
    the same reason as :class:`_Handler`.
    """

    matches: Callable[[Any], bool]
    to_table: Callable[[Any], Any]


def _to_arrow_table(value: Any) -> Any:
    """Convert *value* to an Arrow Table with shape metadata, via the first matching rule.

    Detection used the same rules, so a miss here means registry drift.
    """
    for rule in _ARROW_TYPE_RULES:
        if rule.matches(value):
            return rule.to_table(value)
    raise ValueError(f"Cannot convert {type(value).__name__} to Arrow")


def _matches_pyarrow(value: Any) -> bool:
    import pyarrow as pa

    return isinstance(value, (pa.Table, pa.RecordBatch))


def _table_from_pyarrow(value: Any) -> Any:
    import pyarrow as pa

    # Tagged so a pa.RecordBatch doesn't come back as a DataFrame (the reader's default).
    if isinstance(value, pa.RecordBatch):
        return _stamp_metadata(
            pa.Table.from_batches([value]),
            {_META_SHAPE: _SHAPE_TABLE, _META_SOURCE: _SOURCE_PYARROW_RECORD_BATCH},
        )
    return _stamp_metadata(value, {_META_SHAPE: _SHAPE_TABLE, _META_SOURCE: _SOURCE_PYARROW_TABLE})


def _matches_pandas(value: Any) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False
    return isinstance(value, (pd.DataFrame, pd.Series))


def _table_from_pandas(value: Any) -> Any:
    import pandas as pd
    import pyarrow as pa

    if isinstance(value, pd.DataFrame):
        table = pa.Table.from_pandas(value)
        return _stamp_metadata(
            table, {_META_SHAPE: _SHAPE_TABLE, _META_SOURCE: _SOURCE_PANDAS_DATAFRAME}
        )
    # from_pandas needs a DataFrame; stash the name (maybe None or non-string)
    # so the reader can rebuild the Series.
    frame = value.to_frame()
    table = pa.Table.from_pandas(frame)
    return _stamp_metadata(
        table,
        {
            _META_SHAPE: _SHAPE_TABLE,
            _META_SOURCE: _SOURCE_PANDAS_SERIES,
            _META_PD_NAME: _encode_series_name(value.name),
        },
    )


def _matches_polars(value: Any) -> bool:
    # A polars value implies polars is imported; probing sys.modules avoids
    # importing it just to reject non-polars values.
    pl = sys.modules.get("polars")
    return pl is not None and isinstance(value, (pl.DataFrame, pl.Series))


def _table_from_polars(value: Any) -> Any:
    import polars as pl

    if isinstance(value, pl.DataFrame):
        return _stamp_metadata(
            value.to_arrow(), {_META_SHAPE: _SHAPE_TABLE, _META_SOURCE: _SOURCE_POLARS_DATAFRAME}
        )
    # The polars Series name is always a string and survives as the Arrow field
    # name, so no extra metadata (unlike pandas).
    return _stamp_metadata(
        value.to_frame().to_arrow(),
        {_META_SHAPE: _SHAPE_TABLE, _META_SOURCE: _SOURCE_POLARS_SERIES},
    )


def _matches_numpy(value: Any) -> bool:
    try:
        import numpy as np
    except ImportError:
        return False
    return isinstance(value, (np.ndarray, np.generic))


def _table_from_numpy(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return _ndarray_to_table(value)
    # Round-trips as the Python primitive; the numpy scalar type is lost.
    return _python_scalar_to_table(value.item())


def _matches_torch(value: Any) -> bool:
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(value, torch.Tensor)


def _table_from_torch(value: Any) -> Any:
    # detach() drops autograd (numpy() refuses grad tensors); cpu() copies device
    # tensors back. bfloat16 and sparse raise here and fall back to pickle.
    arr = value.detach().cpu().numpy()
    return _ndarray_to_table(arr, source=_SOURCE_TORCH)


def _matches_jax(value: Any) -> bool:
    jax = sys.modules.get("jax")
    return jax is not None and isinstance(value, jax.Array)


def _table_from_jax(value: Any) -> Any:
    import numpy as np

    arr = np.asarray(value)
    return _ndarray_to_table(arr, source=_SOURCE_JAX)


def _matches_typed_scalar(value: Any) -> bool:
    import datetime as _dt
    from decimal import Decimal
    from uuid import UUID

    return isinstance(
        value,
        (_dt.datetime, _dt.date, _dt.time, _dt.timedelta, Decimal, bytes, bytearray, UUID, complex),
    )


def _table_from_typed_scalar(value: Any) -> Any:
    """Wrap a typed Python primitive in a 1-row scalar Table.

    UUID and complex need a custom representation (binary(16), a struct of floats)
    plus a scalar-type tag; datetime / Decimal / bytes round-trip natively.
    """
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

    # datetime family / Decimal / bytes / bytearray
    return _python_scalar_to_table(value)


def _matches_arrow_capsule(value: Any) -> bool:
    """Anything that exports itself through the Arrow PyCapsule interface.

    The generic table hatch (duckdb, cudf, ibis, ...). It sits below the pandas /
    polars rules, which also implement the capsule but can name their type on the
    way back. The probe is on ``type(value)``: a class object carries its instances'
    protocol methods, and an instance ``hasattr`` can run a ``__getattr__`` that
    raises something other than ``AttributeError``. Exporters that are iterators are
    excluded because reading a one-shot stream consumes it.
    """
    cls = type(value)
    if hasattr(cls, "__next__"):
        return False
    return hasattr(cls, "__arrow_c_stream__")


def _table_from_arrow_capsule(value: Any) -> Any:
    import pyarrow as pa

    # pa.table() runs a lazy handle's query. A value that is also the display
    # output converts twice, so a non-deterministic source stores two datasets.
    return _stamp_metadata(
        pa.table(value), {_META_SHAPE: _SHAPE_TABLE, _META_SOURCE: _SOURCE_ARROW_CAPSULE}
    )


def _matches_dlpack(value: Any) -> bool:
    """Anything that exports a raw n-d buffer through DLPack.

    The generic array hatch (cupy, tensorflow, mlx). ``__dlpack__`` rather than
    ``__array__``: richer types (xarray, astropy Quantity, PIL images) implement
    ``__array__`` as a lossy view, and routing on it would drop what pickle keeps
    and capture PNG display values as tensors. Probed on ``type(value)`` as in
    :func:`_matches_arrow_capsule`.
    """
    return hasattr(type(value), "__dlpack__")


def _table_from_dlpack(value: Any) -> Any:
    import numpy as np

    # Device buffers (CUDA cupy) raise here; the arrow fallback pickles them.
    return _ndarray_to_table(np.from_dlpack(value))


# Rules that know the type and stamp a _META_SOURCE tag to rebuild it. The
# families are disjoint, so order is cheapest-first. torch / jax probe
# sys.modules, costing nothing in notebooks that never import them.
_NAMED_ARROW_TYPE_RULES: tuple[_ArrowRule, ...] = (
    _ArrowRule(_matches_pyarrow, _table_from_pyarrow),
    _ArrowRule(_matches_pandas, _table_from_pandas),
    _ArrowRule(_matches_polars, _table_from_polars),
    _ArrowRule(_matches_numpy, _table_from_numpy),
    _ArrowRule(_matches_torch, _table_from_torch),
    _ArrowRule(_matches_jax, _table_from_jax),
    _ArrowRule(_matches_typed_scalar, _table_from_typed_scalar),
)

# Protocol-only rules (__arrow_c_stream__, __dlpack__) can't name the type on
# read, so they go last: pandas, polars, numpy, torch and jax all export those
# protocols and would be downgraded. Separate tuples make the order structural.
_GENERIC_ARROW_TYPE_RULES: tuple[_ArrowRule, ...] = (
    _ArrowRule(_matches_arrow_capsule, _table_from_arrow_capsule),
    _ArrowRule(_matches_dlpack, _table_from_dlpack),
)

_ARROW_TYPE_RULES: tuple[_ArrowRule, ...] = _NAMED_ARROW_TYPE_RULES + _GENERIC_ARROW_TYPE_RULES


def _python_scalar_to_table(value: Any) -> Any:
    """Wrap a single primitive in a 1-row, 1-column Table via pa.array."""
    import pyarrow as pa

    pa_arr = pa.array([value])
    table = pa.table({"value": pa_arr})
    return _stamp_metadata(table, {_META_SHAPE: _SHAPE_SCALAR})


def _ndarray_to_table(arr: Any, source: bytes | None = None) -> Any:
    """Encode an ndarray as a 1-column Table plus tensor shape metadata.

    *source* records the originating library (torch / jax) so the reader can
    rebuild that type.
    """
    import numpy as np
    import pyarrow as pa

    contiguous = np.ascontiguousarray(arr)
    flat = contiguous.reshape(-1)
    if str(flat.dtype) in _REAL_VIEW_OF_COMPLEX:
        # Arrow has no complex type. Interleaved real/imag keeps it on the tensor
        # path, where _META_TENSOR_DTYPE lets the reader restore it with a view.
        flat = flat.view(_REAL_VIEW_OF_COMPLEX[str(flat.dtype)])
    pa_arr = pa.array(flat)
    table = pa.table({"values": pa_arr})
    meta = {
        _META_SHAPE: _SHAPE_TENSOR,
        _META_TENSOR_SHAPE: json.dumps(list(contiguous.shape)).encode("utf-8"),
        _META_TENSOR_DTYPE: str(contiguous.dtype).encode("utf-8"),
    }
    if source is not None:
        meta[_META_SOURCE] = source
    return _stamp_metadata(table, meta)


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


def to_serialization_safe(value: Any, *, keep_none: bool = False) -> Any:
    """Return a JSON- and TOML-compatible form of *value*.

    The single sanitization boundary for manifest.json, notebook.toml and REST/WS
    payloads: the output holds only ``bool``, ``int``, ``float``, ``str``, ``list``
    and ``dict[str, ...]``. ``None`` becomes ``""`` (TOML has no null) unless
    *keep_none*, for a JSON-only target; sequences and dicts recurse (keys
    stringified), and anything else becomes ``str(value)``.
    """
    if value is None:
        return None if keep_none else ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value
    # numpy 2.0+ scalars don't subclass Python int/float.
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
        return [to_serialization_safe(item, keep_none=keep_none) for item in value]
    if isinstance(value, dict):
        return {str(k): to_serialization_safe(v, keep_none=keep_none) for k, v in value.items()}
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
    filename = f"{_safe_filename_stem(variable_name)}.md"
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

    filename = f"{_safe_filename_stem(variable_name)}.png"
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


# Handlers return ``(png_bytes, width|None, height|None)`` or ``None`` to
# defer. Order: ``_repr_png_``, then matplotlib, then PIL.
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
    """Probe PIL for a PNG's size; ``(None, None)`` if PIL is missing or fails."""
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

    Keeps ``arrow/ipc`` metadata and a ``.arrow`` name so downstream loading treats
    the value as a table; the JSON content carries a marker ``_deserialize_arrow``
    reads even without ``pyarrow``.
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

    filename = f"{_safe_filename_stem(variable_name)}.arrow"
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
    filename = f"{_safe_filename_stem(variable_name)}.json"
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
    filename = f"{_safe_filename_stem(variable_name)}.module.json"
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
    filename = f"{_safe_filename_stem(variable_name)}.cell_instance.pickle"
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
    # Raise rather than return a success-shape dict with file=None: the store
    # would think it was written. harness.py and pool_worker.py turn the
    # exception into an error entry at the producing cell.
    filename = f"{_safe_filename_stem(variable_name)}.pickle"
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


# --- Deserialization ---

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

    *output_dir* is ignored; *file_path* is used as given. *content_type* may be a
    ``ContentType`` or its string.
    """
    handler = _HANDLERS.get(content_type)
    if handler is None or handler.deserialize is None:
        raise ValueError(f"Unknown content type: {content_type!r}")
    return handler.deserialize(Path(file_path))


def _load_arrow_table(blob: bytes) -> Any | None:
    """Open a table-shaped Arrow IPC *blob* as a chunk-consolidated Table.

    Returns ``None`` for tensor/scalar shapes, JSON-fallback blobs and anything
    unreadable as Arrow IPC.
    """
    import pyarrow as pa

    try:
        reader = pa.ipc.open_stream(pa.BufferReader(blob))
        table = reader.read_all()
    except (pa.ArrowInvalid, OSError):
        return None

    meta = table.schema.metadata or {}
    if meta.get(_META_SHAPE, _SHAPE_TABLE) != _SHAPE_TABLE:
        return None
    return table.combine_chunks()


def _coerce_filter_scalar(value: Any, arrow_type: Any) -> Any | None:
    """Coerce a JSON filter *value* to a pyarrow scalar of *arrow_type*.

    Returns ``None`` when the column's type cannot represent it, so the caller skips
    the filter.
    """
    import pyarrow as pa

    try:
        return pa.scalar(value, type=arrow_type)
    except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError, TypeError):
        pass
    if pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type):
        try:
            number = float(value)
        except (ValueError, TypeError):
            return None
        cast = number if pa.types.is_floating(arrow_type) else int(number)
        try:
            return pa.scalar(cast, type=arrow_type)
        except (pa.ArrowInvalid, ValueError, TypeError):
            return None
    if pa.types.is_timestamp(arrow_type) or pa.types.is_date(arrow_type):
        import pandas as pd

        try:
            stamp = pd.Timestamp(value)
        except (ValueError, TypeError):
            return None
        try:
            return pa.scalar(stamp.to_pydatetime(), type=arrow_type)
        except (pa.ArrowInvalid, ValueError, TypeError):
            return None
    return None


def _apply_filters(table: Any, filters: list[dict[str, Any]] | None) -> Any:
    """AND per-column filter predicates (``{col, op, value, value2}``) over the table.

    Unknown columns or ops and uncoercible values are skipped, so a half-typed
    filter never fails the viewer.
    """
    import pyarrow as pa
    import pyarrow.compute as _pc

    # pyarrow.compute registers functions at runtime, unknown to ty's stubs.
    pc = cast(Any, _pc)

    if not filters:
        return table

    mask = None
    for spec in filters:
        col = spec.get("col")
        op = str(spec.get("op") or "")
        if col not in table.column_names:
            continue
        column = table[col]
        ftype = table.schema.field(col).type

        if op == "is_null":
            predicate = pc.is_null(column)
        elif op == "not_null":
            predicate = pc.is_valid(column)
        elif op == "contains":
            try:
                as_str = pc.cast(column, pa.string())
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                continue
            predicate = pc.match_substring(as_str, str(spec.get("value", "")), ignore_case=True)
        elif op == "between":
            low = _coerce_filter_scalar(spec.get("value"), ftype)
            high = _coerce_filter_scalar(spec.get("value2"), ftype)
            if low is None or high is None:
                continue
            predicate = pc.and_(pc.greater_equal(column, low), pc.less_equal(column, high))
        else:
            scalar = _coerce_filter_scalar(spec.get("value"), ftype)
            if scalar is None:
                continue
            comparators = {
                "eq": pc.equal,
                "ne": pc.not_equal,
                "gt": pc.greater,
                "ge": pc.greater_equal,
                "lt": pc.less,
                "le": pc.less_equal,
            }
            fn = comparators.get(op)
            if fn is None:
                continue
            predicate = fn(column, scalar)

        mask = predicate if mask is None else pc.and_(mask, predicate)

    return table if mask is None else table.filter(mask)


def _apply_search(table: Any, search: str | None) -> Any:
    """Keep rows where *search* appears (case-insensitive) in any column.

    Columns that cannot be cast to string (nested types) match nothing.
    """
    import pyarrow as pa
    import pyarrow.compute as _pc

    pc = cast(Any, _pc)  # see _apply_filters: pyarrow.compute members are runtime-registered

    if not search:
        return table

    mask = None
    for col in table.column_names:
        try:
            as_str = pc.cast(table[col], pa.string())
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            continue
        hit = pc.match_substring(as_str, search, ignore_case=True)
        mask = hit if mask is None else pc.or_(mask, hit)

    if mask is None:
        return table
    # match_substring yields null for null cells; treat those as non-matches.
    return table.filter(pc.fill_null(mask, False))


def _filter_search_sort(
    table: Any,
    filters: list[dict[str, Any]] | None,
    search: str | None,
    sort_by: str | None,
    sort_dir: str,
) -> Any:
    """Apply filters, then global search, then a global sort, in that order."""
    table = _apply_filters(table, filters)
    table = _apply_search(table, search)
    if sort_by and sort_by in table.column_names:
        order = "descending" if sort_dir == "desc" else "ascending"
        table = table.sort_by([(sort_by, order)])
    return table


def read_table_page(
    blob: bytes,
    *,
    offset: int = 0,
    limit: int = 100,
    sort_by: str | None = None,
    sort_dir: str = "asc",
    search: str | None = None,
    filters: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Decode an Arrow IPC *blob* into a JSON-safe page of table rows for the viewer.

    Filters, search and sort apply to the whole table before slicing, so ``total``
    is the filtered count and the order is global. Returns ``None`` for a blob that
    is not a table-shaped Arrow IPC stream.
    """
    table = _load_arrow_table(blob)
    if table is None:
        return None

    table = _filter_search_sort(table, filters, search, sort_by, sort_dir)
    total = table.num_rows

    page = table.slice(max(0, offset), max(0, limit))
    columns = list(page.column_names)
    pydict = page.to_pydict()
    rows = [
        [to_serialization_safe(pydict[col][i]) for col in columns] for i in range(page.num_rows)
    ]
    return {"columns": columns, "rows": rows, "total": total}


def read_table_summary(blob: bytes) -> dict[str, Any] | None:
    """Per-column summary of a table-shaped Arrow *blob* for the viewer header.

    Each column has dtype, null count and distinct count, plus min/max for numeric
    and temporal columns. ``None`` for non-table blobs.
    """
    import pyarrow as pa
    import pyarrow.compute as _pc

    pc = cast(Any, _pc)  # see _apply_filters: pyarrow.compute members are runtime-registered

    table = _load_arrow_table(blob)
    if table is None:
        return None

    columns: list[dict[str, Any]] = []
    for name in table.column_names:
        column = table[name]
        ftype = table.schema.field(name).type
        info: dict[str, Any] = {
            "name": name,
            "dtype": str(ftype),
            "nulls": column.null_count,
            "distinct": pc.count_distinct(column).as_py(),
            "min": None,
            "max": None,
        }
        orderable = (
            pa.types.is_integer(ftype)
            or pa.types.is_floating(ftype)
            or pa.types.is_temporal(ftype)
            or pa.types.is_decimal(ftype)
        )
        if orderable and column.null_count < len(column):
            info["min"] = to_serialization_safe(pc.min(column).as_py())
            info["max"] = to_serialization_safe(pc.max(column).as_py())
        columns.append(info)

    return {"columns": columns, "total": table.num_rows}


def write_table_export(
    blob: bytes,
    fmt: str,
    *,
    sort_by: str | None = None,
    sort_dir: str = "asc",
    search: str | None = None,
    filters: list[dict[str, Any]] | None = None,
) -> bytes | None:
    """Serialize a table-shaped Arrow *blob* to CSV or Parquet bytes.

    Applies the viewer's filters, search and sort, so a download matches the
    screen. ``None`` for non-table blobs or an unsupported *fmt*.
    """
    table = _load_arrow_table(blob)
    if table is None:
        return None

    table = _filter_search_sort(table, filters, search, sort_by, sort_dir)
    buffer = io.BytesIO()
    if fmt == "csv":
        import pyarrow.csv as pacsv

        pacsv.write_csv(table, buffer)
    elif fmt == "parquet":
        import pyarrow.parquet as papq

        papq.write_table(table, buffer)
    else:
        return None
    return buffer.getvalue()


def _deserialize_arrow(file_path: Path) -> Any:
    """Read an Arrow IPC stream, rebuilding the type named by ``strata.arrow.shape``."""
    fallback_payload = _read_arrow_json_fallback(file_path)
    if fallback_payload is not None:
        return _deserialize_arrow_json_fallback(fallback_payload)

    import pyarrow as pa

    with open(file_path, "rb") as f:
        reader = pa.ipc.open_stream(f)
        table = reader.read_all()

    # Flat Arrays, not ChunkedArrays: pa.RecordBatch.from_pandas fails on
    # DataFrames that survived an Arrow round-trip otherwise.
    table = table.combine_chunks()

    meta = table.schema.metadata or {}
    shape = meta.get(_META_SHAPE, _SHAPE_TABLE)

    if shape == _SHAPE_TENSOR:
        return _tensor_from_table(table)

    if shape == _SHAPE_SCALAR:
        return _extract_scalar_from_table(table)

    # Default / table shape: pandas DataFrame or pa.Table.
    return _table_to_pandas_or_arrow(table)


def _table_to_pandas_or_arrow(table: Any) -> Any:
    """Decode a shape=table Arrow Table back to pandas or pyarrow.

    If the value came from pandas but ``to_pandas()`` fails, returns the pa.Table
    and logs it, since a silent type change confuses downstream cells.
    """
    meta = table.schema.metadata or {}
    source = meta.get(_META_SOURCE, b"")

    if source in (_SOURCE_PYARROW_TABLE, _SOURCE_ARROW_CAPSULE):
        return table
    if source == _SOURCE_PYARROW_RECORD_BATCH:
        import pyarrow as pa

        # combine_chunks leaves at most one chunk per column: the single stored batch
        # (or none when empty).
        batches = table.combine_chunks().to_batches()
        return batches[0] if batches else pa.RecordBatch.from_pylist([], schema=table.schema)

    if source in (_SOURCE_POLARS_DATAFRAME, _SOURCE_POLARS_SERIES):
        polars_value = _table_to_polars(table, source)
        if polars_value is not None:
            return polars_value
        # polars absent on read: pandas/arrow is a usable view of the same table.

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
            series.name = _decode_series_name(meta.get(_META_PD_NAME, b""))
            return series
        except Exception as exc:
            logger.warning(
                "Reconstructing pandas.Series from Arrow failed (%s); returning DataFrame instead.",
                exc,
            )
            return frame
    return frame


def _table_to_polars(table: Any, source: bytes) -> Any | None:
    """Rebuild a polars DataFrame / Series, or ``None`` if polars is absent.

    The Series name round-trips through the Arrow field name.
    """
    try:
        import polars as pl
    except ImportError:
        return None

    # Always a Table, so a DataFrame; the isinstance narrows for the type checker.
    frame = pl.from_arrow(table)
    if source == _SOURCE_POLARS_SERIES and isinstance(frame, pl.DataFrame):
        return frame.to_series(0)
    return frame


def _tensor_from_table(table: Any) -> Any:
    """Decode a shape=tensor Arrow Table back to its original array type.

    Returns a torch / jax array when ``_META_SOURCE`` names one and that library is
    importable, else a numpy ndarray.
    """
    import numpy as np

    meta = table.schema.metadata or {}
    raw_shape = meta.get(_META_TENSOR_SHAPE, b"[]").decode("utf-8")
    dtype_str = meta.get(_META_TENSOR_DTYPE, b"").decode("utf-8")
    shape = tuple(json.loads(raw_shape))

    flat = table.column(0).to_numpy(zero_copy_only=False)
    if dtype_str:
        target = np.dtype(dtype_str)
        if str(target) in _REAL_VIEW_OF_COMPLEX:
            flat = flat.astype(_REAL_VIEW_OF_COMPLEX[str(target)], copy=False).view(target)
        else:
            flat = flat.astype(target, copy=False)
    arr = np.ascontiguousarray(flat).reshape(shape)

    source = meta.get(_META_SOURCE, b"")
    if source == _SOURCE_TORCH:
        try:
            import torch  # ty: ignore[unresolved-import]  # optional; not in the dev env
        except ImportError:
            return arr
        return torch.from_numpy(arr)
    if source == _SOURCE_JAX:
        try:
            import jax.numpy as jnp  # ty: ignore[unresolved-import]  # optional
        except ImportError:
            return arr
        converted = jnp.asarray(arr)
        if converted.dtype != arr.dtype:
            # A 64-bit jax.Array proves x64 was on where it was made, so enable it here
            # (settable after import) and convert again rather than narrow or refuse.
            # Only widened, and only on that evidence, so a float32 notebook is untouched.
            import jax  # ty: ignore[unresolved-import]  # optional

            jax.config.update("jax_enable_x64", True)
            converted = jnp.asarray(arr)
            _record_x64_enabled()
        if converted.dtype != arr.dtype:
            # x64 didn't repair it: JAX can't represent this dtype. Never narrow silently.
            # Checked on the outcome, not by predicting promotion rules.
            raise StrataPrecisionError(str(arr.dtype), str(converted.dtype))
        return converted
    return arr


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


def _deserialize_file_path(file_path: Path) -> Path:
    return file_path


def _deserialize_rds(file_path: Path) -> Any:
    # Python has no RDS reader; raise the structured error with the suggested fix.
    raise StrataRArtifactError(file_path)


def _deserialize_pickle(file_path: Path) -> Any:
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    codec_payload = _unwrap_codec_payload(data)
    if codec_payload is None:
        # Older artifacts stored raw pickle payloads.
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
    injected: dict[str, Any] | None = None,
):
    import types

    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        module.__file__ = str(file_path)
        sys.modules[module_name] = module
        # Before exec, so module-load references (defaults, bases, decorators)
        # resolve. module_name folds in the injected identity for a distinct cache key.
        if injected:
            module.__dict__.update(injected)
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


def _deserialize_cell_module(file_path: Path, injected: dict[str, Any] | None = None) -> Any:
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    module_name = data.get("module_name")
    symbol_name = data.get("symbol_name")
    module_source = data.get("source")
    if not isinstance(module_name, str) or not isinstance(symbol_name, str):
        raise ValueError("Invalid exported notebook module descriptor")
    if not isinstance(module_source, str):
        raise ValueError(f"Exported notebook module '{module_name}' has invalid source")

    module = _ensure_cell_module(module_name, module_source, file_path, injected=injected)

    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ValueError(
            f"Exported notebook module '{module_name}' does not define '{symbol_name}'"
        ) from exc


def deserialize_cell_module_with_injection(file_path: Path, injected: dict[str, Any]) -> Any:
    """Deserialize a ``module/cell`` export with *injected* upstream values in its namespace."""
    return _deserialize_cell_module(file_path, injected=injected)


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
        # The first module/cell-instance format.
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


# --- Handler registry ---
# ContentType -> (serialize, deserialize). At module bottom so every handler is
# defined. Keyed by ``str`` so enum and raw string content types both resolve.

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
    # Python never produces RDS; harness.R writes it.
    ContentType.RDS_OBJECT: _Handler(None, _deserialize_rds),
    ContentType.FILE_PATH: _Handler(None, _deserialize_file_path),
}
