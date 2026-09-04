"""Tests for serializer module."""

from __future__ import annotations

import contextlib
import json
import logging
import pickle
import sys
import tempfile
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from strata.notebook import Markdown
from strata.notebook import serializer as serializer_module
from strata.notebook.serializer import (
    EXT_TO_CONTENT_TYPE,
    ContentType,
    StrataRArtifactError,
    deserialize_value,
    read_table_page,
    serialize_value,
)

_MINIMAL_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x04\x00\x00\x00\xb5\x1c\x0c\x02\x00\x00\x00\x0bIDATx\xdac\xfc\xff"
    b"\x1f\x00\x03\x03\x02\x00\xef\x9b\xe0M\x00\x00\x00\x00IEND\xaeB`\x82"
)


# Module-level classes for pickle tests (local classes can't be pickled)
class _PickleTestCustomClass:
    def __init__(self, x):
        self.x = x


class _PickleTestMyModel:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __eq__(self, other):
        return self.name == other.name and self.value == other.value


class _SerializerNoStatePerson:
    name = "John"
    age = 20

    def __str__(self):
        return f"{self.name}:{self.age}"


class _SerializerSlotPerson:
    __slots__ = ("name", "age")

    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def __str__(self):
        return f"{self.name}:{self.age}"


class _SerializerBaseSlotPerson:
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name


class _SerializerDerivedSlotPerson(_SerializerBaseSlotPerson):
    __slots__ = ("age",)

    def __init__(self, name: str, age: int):
        super().__init__(name)
        self.age = age

    def __str__(self):
        return f"{self.name}:{self.age}"


class _SerializerCustomStatePerson:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
        self.restored = False

    def __getstate__(self):
        return {"payload": f"{self.name}|{self.age}"}

    def __setstate__(self, state):
        self.name, age = state["payload"].split("|")
        self.age = int(age)
        self.restored = True

    def __str__(self):
        return f"{self.name}:{self.age}:{self.restored}"


class _SerializerPngDisplay:
    def _repr_png_(self):
        return _MINIMAL_PNG_BYTES


class _SerializerMarkdownDisplay:
    def _repr_markdown_(self):
        return "# Title\n\n- one\n- two"


@pytest.fixture(autouse=True)
def _undo_cell_module_marking():
    """Strip the cell-module marks the round-trip tests leave on this module.

    ``_mark_as_cell_module`` stamps *this test module* as a cell module, and
    deserializing a ``module/cell-instance`` artifact then walks that module
    and sets ``__strata_cell_exported_class__`` on every class defined in it
    (``serializer.py`` ``_load_cell_module``) — not just the one under test.
    Left behind, every module-level class in this file is detected as a cell
    instance for the rest of the session, so an unrelated test asserting
    ``pickle/object`` passes or fails depending on what ran before it.
    """
    yield
    module = sys.modules[__name__]
    module.__dict__.pop("__strata_cell_module__", None)
    module.__dict__.pop("__strata_cell_module_source__", None)
    for value in list(module.__dict__.values()):
        # Mirrors the ``__module__`` check in ``_load_cell_module``: imported
        # names (pd.DataFrame, pa.Table) live in this dict too and were never
        # stamped, and extension types refuse delattr outright.
        if isinstance(value, type) and getattr(value, "__module__", None) == __name__:
            try:
                delattr(value, "__strata_cell_exported_class__")
            except AttributeError:
                continue


def _mark_as_cell_module(cls, module_source: str) -> None:
    module = sys.modules[cls.__module__]
    module.__dict__["__strata_cell_module__"] = True
    module.__dict__["__strata_cell_module_source__"] = module_source
    setattr(cls, "__strata_cell_exported_class__", True)


class TestArrowSerialization:
    """Test Arrow IPC serialization."""

    def test_serialize_dataframe(self):
        """Test serializing a pandas DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

        with tempfile.TemporaryDirectory() as tmpdir:
            result = serialize_value(df, Path(tmpdir), "df")

            assert result["content_type"] == "arrow/ipc"
            assert result["rows"] == 3
            assert result["columns"] == ["a", "b"]
            assert result["bytes"] > 0
            assert result["preview"] == [[1, 4.0], [2, 5.0], [3, 6.0]]

    def test_serialize_arrow_table(self):
        """Test serializing a PyArrow Table."""
        table = pa.table({"x": [10, 20, 30], "y": ["a", "b", "c"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            result = serialize_value(table, Path(tmpdir), "tbl")

            assert result["content_type"] == "arrow/ipc"
            assert result["rows"] == 3
            assert result["columns"] == ["x", "y"]

    def test_roundtrip_pyarrow_table_stays_a_table(self):
        """A pa.Table handed to the next cell must still be a pa.Table.

        Every other table source stamps ``strata.arrow.source`` so the reader
        can reconstruct the exact type; pyarrow values stamped only the shape,
        so they fell through to the reader's ``to_pandas()`` default. A cell
        that did ``t.column("a")`` on its upstream's table got
        ``AttributeError: 'DataFrame' object has no attribute 'column'``.
        """
        table = pa.table({"a": [1, 2], "b": ["x", "y"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            meta = serialize_value(table, tmpdir, "tbl")
            back = deserialize_value(meta["content_type"], tmpdir / meta["file"])

        assert isinstance(back, pa.Table)
        assert back.equals(table)

    def test_roundtrip_pyarrow_record_batch_stays_a_record_batch(self):
        batch = pa.RecordBatch.from_pydict({"a": [1, 2]})

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            meta = serialize_value(batch, tmpdir, "batch")
            back = deserialize_value(meta["content_type"], tmpdir / meta["file"])

        assert isinstance(back, pa.RecordBatch)
        assert back.equals(batch)

    def test_roundtrip_empty_pyarrow_values(self):
        """An empty table has no batches to take, so the RecordBatch path has
        to rebuild one from the schema rather than index into an empty list."""
        table = pa.table({"a": pa.array([], type=pa.int64())})
        batch = pa.RecordBatch.from_pydict({"a": pa.array([], type=pa.int64())})

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for name, value in (("t", table), ("b", batch)):
                meta = serialize_value(value, tmpdir, name)
                back = deserialize_value(meta["content_type"], tmpdir / meta["file"])
                assert type(back) is type(value)
                assert back.equals(value)

    def test_an_untagged_table_still_reads_back_as_pandas(self):
        """Artifacts written before the source tag existed — and core scan
        results, which carry no source — must keep their pandas behaviour."""
        from strata.notebook.serializer import (
            _SHAPE_TABLE,
            _stamp_shape,
            _table_to_pandas_or_arrow,
        )

        legacy = _stamp_shape(pa.table({"a": [1, 2]}), _SHAPE_TABLE)
        assert isinstance(_table_to_pandas_or_arrow(legacy), pd.DataFrame)
        assert isinstance(_table_to_pandas_or_arrow(pa.table({"a": [1]})), pd.DataFrame)

    def test_json_encoding_is_not_used_when_it_would_change_the_value(self):
        """``json.dumps`` coerces rather than failing, so the encode probe
        can't see these.

        Non-string dict keys become strings and tuples become lists, so a cell
        that stored ``{1: "a"}`` handed the next one ``{"1": "a"}`` and a
        ``counts[1]`` downstream raised KeyError. Both now take the pickle
        path, which preserves them.
        """
        lossy = [
            {1: "a", 2: "b"},
            {1.5: "a"},
            {True: "a"},
            {"outer": {2: "b"}},
            {"a": (1, 2)},
            [(1, 2)],
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for index, value in enumerate(lossy):
                meta = serialize_value(value, tmpdir, f"v{index}")
                back = deserialize_value(meta["content_type"], tmpdir / meta["file"])
                assert back == value, f"{value!r} did not survive the round trip"
                assert type(back) is type(value)

    def test_json_stays_the_encoding_for_values_it_preserves(self):
        """The guard must not push ordinary values onto the pickle path.

        NaN and Inf matter here: the JSON writer round-trips both, so they are
        deliberately not treated as losses even though ``nan != nan`` would
        make an equality-based probe say otherwise.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for index, value in enumerate([{"a": [1, {"b": 2}]}, [1, 2, 3], {"x": float("inf")}]):
                meta = serialize_value(value, tmpdir, f"keep{index}")
                assert meta["content_type"] == "json/object", value

            nan_meta = serialize_value({"x": float("nan")}, tmpdir, "nan")
            assert nan_meta["content_type"] == "json/object"
            back = deserialize_value(nan_meta["content_type"], tmpdir / nan_meta["file"])
            assert back["x"] != back["x"]  # still NaN

    def test_roundtrip_dataframe(self):
        """Test round-trip: serialize and deserialize a DataFrame."""
        df_orig = pd.DataFrame({"id": [1, 2, 3], "value": [1.5, 2.5, 3.5], "name": ["a", "b", "c"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Serialize
            meta = serialize_value(df_orig, tmpdir, "data")
            file_path = tmpdir / meta["file"]

            # Deserialize
            df_loaded = deserialize_value(meta["content_type"], file_path)

            # Convert to pandas for comparison
            if isinstance(df_loaded, pa.Table):
                df_loaded = df_loaded.to_pandas()

            # Check shape and values
            assert df_loaded.shape == df_orig.shape
            assert list(df_loaded.columns) == list(df_orig.columns)
            pd.testing.assert_frame_equal(df_loaded, df_orig)

    def test_serialize_arrow_with_nulls(self):
        """Test Arrow serialization with null values."""
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, 2.0, 3.0]})

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            meta = serialize_value(df, tmpdir, "nulls")
            file_path = tmpdir / meta["file"]

            result = deserialize_value(meta["content_type"], file_path)
            if isinstance(result, pa.Table):
                result = result.to_pandas()

            # Verify nulls are preserved
            assert result.iloc[0, 0] == 1
            assert pd.isna(result.iloc[1, 0])
            assert pd.isna(result.iloc[0, 1])

    def test_arrow_json_fallback_roundtrips_dataframe_after_pyarrow_error(self, monkeypatch):
        """PyArrow conversion errors should fall back to a JSON-backed table artifact."""
        df = pd.DataFrame(
            {
                "when": [date(2024, 1, 2), date(2024, 1, 3)],
                "amount": [Decimal("1.25"), Decimal("2.50")],
            }
        )

        def _raise_arrow_invalid(value, output_dir, variable_name):
            raise pa.ArrowInvalid("unsupported dtype")

        monkeypatch.setattr(serializer_module, "_serialize_arrow", _raise_arrow_invalid)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            meta = serialize_value(df, tmpdir, "data")
            file_path = tmpdir / meta["file"]

            assert meta["content_type"] == "arrow/ipc"
            assert meta["file"] == "data.arrow"
            assert meta["columns"] == ["when", "amount"]
            assert meta["preview"] == [["2024-01-02", "1.25"], ["2024-01-03", "2.50"]]

            payload = json.loads(file_path.read_text(encoding="utf-8"))
            assert payload["__strata_arrow_json_fallback__"] is True
            assert payload["kind"] == "dataframe"

            loaded = deserialize_value(meta["content_type"], file_path)
            assert isinstance(loaded, pd.DataFrame)
            assert list(loaded.columns) == ["when", "amount"]
            assert loaded.to_dict(orient="records") == [
                {"when": "2024-01-02", "amount": "1.25"},
                {"when": "2024-01-03", "amount": "2.50"},
            ]

    def test_arrow_json_fallback_roundtrips_series(self, monkeypatch):
        """Series should keep Series shape and name through the JSON fallback path."""
        series = pd.Series([10, 20, 30], name="target")

        def _raise_arrow_value_error(value, output_dir, variable_name):
            raise ValueError("force JSON fallback")

        monkeypatch.setattr(serializer_module, "_serialize_arrow", _raise_arrow_value_error)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            meta = serialize_value(series, tmpdir, "target")
            loaded = deserialize_value(meta["content_type"], tmpdir / meta["file"])

            assert isinstance(loaded, pd.Series)
            assert loaded.name == "target"
            assert loaded.tolist() == [10, 20, 30]

    def test_deserialize_arrow_json_fallback_without_pyarrow(self, monkeypatch):
        """JSON-backed Arrow fallbacks should remain readable even if pyarrow is unavailable."""
        fallback_path = None

        df = pd.DataFrame({"label": ["a", "b"]})

        def _raise_arrow_value_error(value, output_dir, variable_name):
            raise ValueError("force JSON fallback")

        monkeypatch.setattr(serializer_module, "_serialize_arrow", _raise_arrow_value_error)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            meta = serialize_value(df, tmpdir, "labels")
            fallback_path = tmpdir / meta["file"]

            import builtins

            real_import = builtins.__import__

            def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "pyarrow":
                    raise ImportError("pyarrow unavailable")
                return real_import(name, globals, locals, fromlist, level)

            monkeypatch.setattr(builtins, "__import__", _blocked_import)
            loaded = deserialize_value(meta["content_type"], fallback_path)

            assert isinstance(loaded, pd.DataFrame)
            assert loaded.to_dict(orient="records") == [{"label": "a"}, {"label": "b"}]


class TestJsonSerialization:
    """Test JSON serialization."""

    def test_serialize_dict(self):
        """Test serializing a dictionary."""
        data = {"x": 1, "y": "hello", "z": [1, 2, 3]}

        with tempfile.TemporaryDirectory() as tmpdir:
            result = serialize_value(data, Path(tmpdir), "data")

            assert result["content_type"] == "json/object"
            assert result["bytes"] > 0
            assert result["preview"] == data

    def test_serialize_list(self):
        """Test serializing a list."""
        data = [1, 2, 3, "hello"]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = serialize_value(data, Path(tmpdir), "lst")

            assert result["content_type"] == "json/object"
            assert result["preview"] == data

    def test_serialize_scalar(self):
        """Test serializing scalar values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Integer
            result = serialize_value(42, Path(tmpdir), "int_val")
            assert result["content_type"] == "json/object"

            # String
            result = serialize_value("hello", Path(tmpdir), "str_val")
            assert result["content_type"] == "json/object"

            # Boolean
            result = serialize_value(True, Path(tmpdir), "bool_val")
            assert result["content_type"] == "json/object"

    def test_roundtrip_dict(self):
        """Test round-trip for dictionary."""
        data_orig = {"a": 1, "b": "test", "c": [1, 2, 3], "d": None}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            meta = serialize_value(data_orig, tmpdir, "data")
            file_path = tmpdir / meta["file"]
            data_loaded = deserialize_value(meta["content_type"], file_path)

            assert data_loaded == data_orig


class TestImageSerialization:
    """Test PNG display serialization."""

    def test_serialize_repr_png_value(self):
        """Values exposing _repr_png_ should serialize as image/png."""
        value = _SerializerPngDisplay()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = serialize_value(value, Path(tmpdir), "_")

            assert result["content_type"] == "image/png"
            assert result["bytes"] > 0
            assert result["inline_data_url"].startswith("data:image/png;base64,")

    def test_serialize_repr_markdown_value(self):
        """Values exposing _repr_markdown_ should serialize as text/markdown."""
        value = _SerializerMarkdownDisplay()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = serialize_value(value, Path(tmpdir), "_")

            assert result["content_type"] == "text/markdown"
            assert result["bytes"] > 0
            assert result["markdown_text"] == "# Title\n\n- one\n- two"

    def test_serialize_markdown_helper(self):
        """The public Markdown helper should opt into markdown display."""
        value = Markdown("## Heading")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = serialize_value(value, Path(tmpdir), "_")
            file_path = Path(tmpdir) / result["file"]

            assert result["content_type"] == "text/markdown"
            assert deserialize_value(result["content_type"], file_path) == "## Heading"

    def test_roundtrip_nested(self):
        """Test round-trip for nested structure."""
        data_orig = {
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ],
            "count": 2,
            "active": True,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            meta = serialize_value(data_orig, tmpdir, "nested")
            file_path = tmpdir / meta["file"]
            data_loaded = deserialize_value(meta["content_type"], file_path)

            assert data_loaded == data_orig


class TestPickleSerialization:
    """Test pickle serialization."""

    def test_serialize_custom_object(self):
        """Test serializing a custom object."""
        obj = _PickleTestCustomClass(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = serialize_value(obj, Path(tmpdir), "obj")

            assert result["content_type"] == "pickle/object"
            # cloudpickle is the default codec (strict superset of
            # stdlib pickle); stdlib "pickle" is available as an opt-in
            # via STRATA_NOTEBOOK_OBJECT_CODEC.
            assert result["codec"] in {"cloudpickle", "pickle"}
            assert result["type"] == "_PickleTestCustomClass"
            assert result["bytes"] > 0

    def test_roundtrip_custom_object(self):
        """Test round-trip for custom object."""
        obj_orig = _PickleTestMyModel("test", 123)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            meta = serialize_value(obj_orig, tmpdir, "model")
            file_path = tmpdir / meta["file"]
            obj_loaded = deserialize_value(meta["content_type"], file_path)

            assert obj_loaded == obj_orig

    def test_pickle_serialization_uses_codec_envelope(self):
        """Pickle/object files should store a codec envelope for future backends."""
        obj = _PickleTestCustomClass(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            meta = serialize_value(obj, tmpdir, "obj")
            file_path = tmpdir / meta["file"]

            with open(file_path, "rb") as f:
                payload = pickle.load(f)

            assert payload["__strata_object_codec__"] == "strata.notebook.object_codec.v1"
            # Default codec is now cloudpickle; "pickle" is opt-in.
            assert payload["codec"] in {"cloudpickle", "pickle"}
            assert isinstance(payload["payload"], bytes)

    def test_deserialize_legacy_raw_pickle(self):
        """Legacy raw-pickle files should remain readable after codec abstraction."""
        obj = _PickleTestMyModel("legacy", 7)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            file_path = tmpdir / "legacy.pickle"
            with open(file_path, "wb") as f:
                pickle.dump(obj, f, protocol=5)

            loaded = deserialize_value("pickle/object", file_path)
            assert loaded == obj

    def test_roundtrip_cell_instance_without_instance_state(self):
        """module/cell-instance should restore plain class-var instances with no __dict__ state."""
        person = _SerializerNoStatePerson()
        _mark_as_cell_module(
            _SerializerNoStatePerson,
            "class _SerializerNoStatePerson:\n"
            "    name = 'John'\n"
            "    age = 20\n"
            "\n"
            "    def __str__(self):\n"
            '        return f"{self.name}:{self.age}"\n',
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            meta = serialize_value(person, tmpdir, "p")
            assert meta["content_type"] == "module/cell-instance"

            loaded = deserialize_value(meta["content_type"], tmpdir / meta["file"])
            assert str(loaded) == "John:20"

    def test_roundtrip_cell_instance_with_slots(self):
        """module/cell-instance should preserve slot-only instance state."""
        person = _SerializerSlotPerson("Ada", 10)
        _mark_as_cell_module(
            _SerializerSlotPerson,
            "class _SerializerSlotPerson:\n"
            "    __slots__ = ('name', 'age')\n"
            "\n"
            "    def __init__(self, name, age):\n"
            "        self.name = name\n"
            "        self.age = age\n"
            "\n"
            "    def __str__(self):\n"
            '        return f"{self.name}:{self.age}"\n',
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            meta = serialize_value(person, tmpdir, "p")
            loaded = deserialize_value(meta["content_type"], tmpdir / meta["file"])

            assert str(loaded) == "Ada:10"

    def test_roundtrip_cell_instance_with_inherited_slots(self):
        """module/cell-instance should preserve slots defined across base classes."""
        person = _SerializerDerivedSlotPerson("Grace", 30)
        _mark_as_cell_module(
            _SerializerDerivedSlotPerson,
            "class _SerializerBaseSlotPerson:\n"
            "    __slots__ = ('name',)\n"
            "\n"
            "    def __init__(self, name):\n"
            "        self.name = name\n"
            "\n"
            "class _SerializerDerivedSlotPerson(_SerializerBaseSlotPerson):\n"
            "    __slots__ = ('age',)\n"
            "\n"
            "    def __init__(self, name, age):\n"
            "        super().__init__(name)\n"
            "        self.age = age\n"
            "\n"
            "    def __str__(self):\n"
            '        return f"{self.name}:{self.age}"\n',
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            meta = serialize_value(person, tmpdir, "p")
            loaded = deserialize_value(meta["content_type"], tmpdir / meta["file"])

            assert str(loaded) == "Grace:30"

    def test_roundtrip_cell_instance_with_custom_state_methods(self):
        """module/cell-instance should respect custom __getstate__/__setstate__."""
        person = _SerializerCustomStatePerson("Lin", 41)
        _mark_as_cell_module(
            _SerializerCustomStatePerson,
            "class _SerializerCustomStatePerson:\n"
            "    def __init__(self, name, age):\n"
            "        self.name = name\n"
            "        self.age = age\n"
            "        self.restored = False\n"
            "\n"
            "    def __getstate__(self):\n"
            "        return {'payload': f'{self.name}|{self.age}'}\n"
            "\n"
            "    def __setstate__(self, state):\n"
            "        self.name, age = state['payload'].split('|')\n"
            "        self.age = int(age)\n"
            "        self.restored = True\n"
            "\n"
            "    def __str__(self):\n"
            '        return f"{self.name}:{self.age}:{self.restored}"\n',
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            meta = serialize_value(person, tmpdir, "p")
            loaded = deserialize_value(meta["content_type"], tmpdir / meta["file"])

            assert str(loaded) == "Lin:41:True"

    def test_serialize_unpicklable_returns_error(self):
        """Test that unpicklable objects return an error result."""

        # Lambdas defined locally can't be pickled
        def func(x):
            return x + 1

        with tempfile.TemporaryDirectory() as tmpdir:
            result = serialize_value(func, Path(tmpdir), "func")

            # Should return error metadata instead of crashing
            assert result.get("error") is not None or result["content_type"] == "pickle/object"

    def test_deserialize_invalid_cell_module_descriptor(self):
        """Corrupted module/cell descriptors should fail with a clear error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            file_path = tmpdir / "broken.cell_module.json"
            file_path.write_text(
                json.dumps({"module_name": "broken", "source": "x = 1"}),
                encoding="utf-8",
            )

            with pytest.raises(ValueError, match="Invalid exported notebook module descriptor"):
                deserialize_value("module/cell", file_path)

    def test_deserialize_missing_cell_module_symbol(self):
        """Missing exported symbol names should raise a clear error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            file_path = tmpdir / "broken.cell_module.json"
            file_path.write_text(
                json.dumps(
                    {
                        "module_name": "broken_symbol_module",
                        "symbol_name": "missing",
                        "source": "x = 1",
                    }
                ),
                encoding="utf-8",
            )

            with pytest.raises(ValueError, match="does not define 'missing'"):
                deserialize_value("module/cell", file_path)

    def test_deserialize_invalid_cell_instance_payload(self):
        """Corrupted module/cell-instance payloads should fail clearly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            file_path = tmpdir / "broken.cell_instance.pickle"
            with open(file_path, "wb") as f:
                pickle.dump({"module_name": "broken"}, f, protocol=5)

            with pytest.raises(ValueError, match="Invalid notebook-exported instance descriptor"):
                deserialize_value("module/cell-instance", file_path)

    def test_deserialize_invalid_cell_instance_state_payload(self):
        """Invalid codec-tagged state payloads should fail clearly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            file_path = tmpdir / "broken.cell_instance.pickle"
            with open(file_path, "wb") as f:
                pickle.dump(
                    {
                        "module_name": "broken_state_module",
                        "class_name": "_SerializerNoStatePerson",
                        "source": (
                            "class _SerializerNoStatePerson:\n    name = 'John'\n    age = 20\n"
                        ),
                        "state_codec": 123,
                        "state_payload": b"broken",
                    },
                    f,
                    protocol=5,
                )

            with pytest.raises(
                ValueError,
                match="Invalid notebook-exported instance state payload",
            ):
                deserialize_value("module/cell-instance", file_path)


class TestContentTypeDetection:
    """Test content type detection."""

    def test_detect_dataframe(self):
        """Detect DataFrame as arrow/ipc."""
        from strata.notebook.serializer import detect_content_type

        df = pd.DataFrame({"a": [1, 2, 3]})
        assert detect_content_type(df) == "arrow/ipc"

    def test_detect_arrow_table(self):
        """Detect Arrow Table as arrow/ipc."""
        from strata.notebook.serializer import detect_content_type

        table = pa.table({"a": [1, 2, 3]})
        assert detect_content_type(table) == "arrow/ipc"

    def test_detect_dict(self):
        """Detect dict as json/object."""
        from strata.notebook.serializer import detect_content_type

        assert detect_content_type({"a": 1}) == "json/object"

    def test_detect_list(self):
        """Detect list as json/object."""
        from strata.notebook.serializer import detect_content_type

        assert detect_content_type([1, 2, 3]) == "json/object"

    def test_detect_scalar(self):
        """Detect scalars as json/object."""
        from strata.notebook.serializer import detect_content_type

        assert detect_content_type(42) == "json/object"
        assert detect_content_type("hello") == "json/object"
        assert detect_content_type(True) == "json/object"

    def test_detect_custom_object(self):
        """Detect custom object as pickle/object."""
        from strata.notebook.serializer import detect_content_type

        class MyClass:
            pass

        assert detect_content_type(MyClass()) == "pickle/object"

    def test_detect_ndarray_goes_to_arrow_ipc(self):
        """Unified codec: ndarray detects as arrow/ipc, not tensor/arrow."""
        import numpy as np

        from strata.notebook.serializer import detect_content_type

        assert detect_content_type(np.array([1, 2, 3])) == "arrow/ipc"
        assert detect_content_type(np.zeros((2, 3))) == "arrow/ipc"

    def test_detect_numpy_scalar_goes_to_arrow_ipc(self):
        """numpy scalars route through the typed arrow/ipc codec."""
        import numpy as np

        from strata.notebook.serializer import detect_content_type

        assert detect_content_type(np.int64(42)) == "arrow/ipc"
        assert detect_content_type(np.float64(1.5)) == "arrow/ipc"

    def test_detect_typed_primitives_go_to_arrow_ipc(self):
        """Typed Python primitives route through arrow/ipc for fidelity."""
        from datetime import datetime, timedelta
        from uuid import uuid4

        from strata.notebook.serializer import detect_content_type

        assert detect_content_type(datetime.now()) == "arrow/ipc"
        assert detect_content_type(date.today()) == "arrow/ipc"
        assert detect_content_type(timedelta(seconds=1)) == "arrow/ipc"
        assert detect_content_type(Decimal("3.14")) == "arrow/ipc"
        assert detect_content_type(b"bytes") == "arrow/ipc"
        assert detect_content_type(uuid4()) == "arrow/ipc"
        assert detect_content_type(1 + 2j) == "arrow/ipc"


class TestUnifiedArrowCodec:
    """Round-trip tests for the unified arrow/ipc content type."""

    def test_roundtrip_ndarray_1d(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            arr = np.array([1, 2, 3, 4, 5])
            meta = serialize_value(arr, tmp, "x")
            assert meta["content_type"] == "arrow/ipc"
            back = deserialize_value(meta["content_type"], Path(tmp) / meta["file"])
            assert isinstance(back, np.ndarray)
            assert back.shape == arr.shape
            assert back.dtype == arr.dtype
            assert (back == arr).all()

    def test_roundtrip_ndarray_multidim(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            meta = serialize_value(arr, tmp, "x")
            assert meta["content_type"] == "arrow/ipc"
            back = deserialize_value(meta["content_type"], Path(tmp) / meta["file"])
            assert back.shape == (2, 3, 4)
            assert back.dtype == np.float32
            assert (back == arr).all()

    def test_roundtrip_numpy_scalar(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            meta = serialize_value(np.int64(42), tmp, "x")
            back = deserialize_value(meta["content_type"], Path(tmp) / meta["file"])
            # Numpy flavor is lost on round-trip; equality is preserved.
            assert back == 42

    def test_roundtrip_datetime(self):
        from datetime import datetime

        with tempfile.TemporaryDirectory() as tmp:
            dt = datetime(2026, 4, 16, 12, 34, 56)
            meta = serialize_value(dt, tmp, "x")
            back = deserialize_value(meta["content_type"], Path(tmp) / meta["file"])
            assert isinstance(back, datetime)
            assert back == dt

    def test_roundtrip_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = date(2026, 4, 16)
            meta = serialize_value(d, tmp, "x")
            back = deserialize_value(meta["content_type"], Path(tmp) / meta["file"])
            assert back == d

    def test_roundtrip_decimal(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = Decimal("3.14159")
            meta = serialize_value(v, tmp, "x")
            back = deserialize_value(meta["content_type"], Path(tmp) / meta["file"])
            assert isinstance(back, Decimal)
            assert back == v

    def test_roundtrip_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = b"hello world"
            meta = serialize_value(v, tmp, "x")
            back = deserialize_value(meta["content_type"], Path(tmp) / meta["file"])
            assert back == v

    def test_roundtrip_uuid(self):
        from uuid import UUID, uuid4

        with tempfile.TemporaryDirectory() as tmp:
            v = uuid4()
            meta = serialize_value(v, tmp, "x")
            back = deserialize_value(meta["content_type"], Path(tmp) / meta["file"])
            assert isinstance(back, UUID)
            assert back == v

    def test_roundtrip_complex(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = 3.0 + 4.0j
            meta = serialize_value(v, tmp, "x")
            back = deserialize_value(meta["content_type"], Path(tmp) / meta["file"])
            assert isinstance(back, complex)
            assert back == v

    def test_roundtrip_dataframe_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
            meta = serialize_value(df, tmp, "x")
            back = deserialize_value(meta["content_type"], Path(tmp) / meta["file"])
            assert isinstance(back, pd.DataFrame)
            assert back.equals(df)

    def test_roundtrip_series_preserves_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = pd.Series([1, 2, 3], name="my_series")
            meta = serialize_value(s, tmp, "x")
            back = deserialize_value(meta["content_type"], Path(tmp) / meta["file"])
            assert isinstance(back, pd.Series)
            assert back.name == "my_series"
            assert list(back) == [1, 2, 3]

    def test_roundtrip_series_falsy_and_typed_names(self):
        """`str(name or "")` collapsed falsy names (0, "", False) to None and
        stringified ints — the exact name (value AND type) must survive."""
        for name in (0, "", False, 5, None, 2.5):
            with tempfile.TemporaryDirectory() as tmp:
                s = pd.Series([1, 2], name=name)
                meta = serialize_value(s, tmp, "x")
                back = deserialize_value(meta["content_type"], Path(tmp) / meta["file"])
                assert isinstance(back, pd.Series)
                assert back.name == name and type(back.name) is type(name), (
                    f"name {name!r} round-tripped as {back.name!r}"
                )


class TestLargeDataFrames:
    """Test serialization of larger DataFrames."""

    def test_serialize_large_dataframe(self):
        """Test serializing a larger DataFrame (1000 rows)."""
        df = pd.DataFrame(
            {
                "id": range(1000),
                "value": [float(i) * 1.5 for i in range(1000)],
                "category": ["A", "B", "C"] * 333 + ["A"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            meta = serialize_value(df, tmpdir, "large")

            assert meta["content_type"] == "arrow/ipc"
            assert meta["rows"] == 1000
            # Preview should only have first 20 rows
            assert len(meta["preview"]) == 20

            # Verify round-trip
            file_path = tmpdir / meta["file"]
            df_loaded = deserialize_value(meta["content_type"], file_path)
            if isinstance(df_loaded, pa.Table):
                df_loaded = df_loaded.to_pandas()

            pd.testing.assert_frame_equal(df_loaded, df)


class TestRdsArtifactRefusal:
    """``application/x-r-rds`` artifacts are R-only — Python deserialization must fail loudly.

    The harness.R fallback tier produces RDS blobs (saveRDS) for any
    value that isn't a data.frame/tibble or a JSON-able scalar/list.
    Python has no RDS reader, so the dispatcher's job is to surface a
    structured error pointing the user back to R for an Arrow re-export
    instead of throwing a confusing "Unknown content type" or returning
    raw bytes.
    """

    def test_content_type_registered(self):
        # The ContentType enum and the file-extension table both need
        # the RDS entry so executor._store_outputs's ``.rds`` ingestion
        # path and the deserializer's dispatch table stay in sync.
        assert ContentType.RDS_OBJECT == "application/x-r-rds"
        assert EXT_TO_CONTENT_TYPE[".rds"] == ContentType.RDS_OBJECT

    def test_deserialize_rds_raises_structured_error(self, tmp_path):
        rds_path = tmp_path / "model.rds"
        rds_path.write_bytes(b"\x1f\x8b\x08\x00fakerds")

        with pytest.raises(StrataRArtifactError) as excinfo:
            deserialize_value(ContentType.RDS_OBJECT, rds_path)

        err = excinfo.value
        assert err.code == "R_ONLY_ARTIFACT"
        assert err.file_path == rds_path
        # The default message (no variable name yet — that's filled in
        # by the Python harness layer) points the user at the fix.
        assert "saveRDS" in str(err)
        assert "data.frame" in str(err)

    def test_deserialize_rds_raises_via_raw_string_content_type(self, tmp_path):
        # Callers from outside the notebook package pass the raw
        # content-type string, not the enum. Both forms must dispatch
        # to the same handler.
        rds_path = tmp_path / "obj.rds"
        rds_path.write_bytes(b"rds")

        with pytest.raises(StrataRArtifactError):
            deserialize_value("application/x-r-rds", rds_path)

    def test_error_carries_variable_name_when_set(self, tmp_path):
        rds_path = tmp_path / "fit.rds"
        rds_path.write_bytes(b"rds")

        err = StrataRArtifactError(rds_path, variable_name="fit")

        assert err.variable_name == "fit"
        assert "fit" in str(err)


class TestPolarsSerialization:
    """polars DataFrames / Series route through the unified arrow/ipc codec
    and round-trip back to polars (degrading to pandas when polars is absent)."""

    def test_detect_polars_dataframe_goes_to_arrow_ipc(self):
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

        with tempfile.TemporaryDirectory() as tmpdir:
            result = serialize_value(df, Path(tmpdir), "df")

        assert result["content_type"] == ContentType.ARROW_IPC
        assert result["rows"] == 3
        assert result["columns"] == ["a", "b"]

    def test_roundtrip_polars_dataframe(self):
        pl = pytest.importorskip("polars")
        orig = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            meta = serialize_value(orig, tmpdir, "data")
            loaded = deserialize_value(meta["content_type"], tmpdir / meta["file"])

        assert isinstance(loaded, pl.DataFrame)
        assert loaded.equals(orig)

    def test_roundtrip_polars_series_preserves_name(self):
        pl = pytest.importorskip("polars")
        orig = pl.Series("temps", [10, 20, 30])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            meta = serialize_value(orig, tmpdir, "s")
            loaded = deserialize_value(meta["content_type"], tmpdir / meta["file"])

        assert isinstance(loaded, pl.Series)
        assert loaded.name == "temps"
        assert loaded.to_list() == [10, 20, 30]

    def test_polars_degrades_to_pandas_when_polars_absent(self, monkeypatch):
        """A polars-sourced artifact stays readable as pandas when the reader
        has no polars — better than pickle, which would need polars."""
        pl = pytest.importorskip("polars")
        orig = pl.DataFrame({"a": [1, 2], "b": [3, 4]})

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            meta = serialize_value(orig, tmpdir, "data")
            # Simulate polars missing on the read side.
            monkeypatch.setitem(__import__("sys").modules, "polars", None)
            loaded = deserialize_value(meta["content_type"], tmpdir / meta["file"])

        assert isinstance(loaded, pd.DataFrame)
        assert list(loaded.columns) == ["a", "b"]


# Real torch / jax are too heavy for the CI matrix, so these exercise the
# detection + codec plumbing through API-compatible stub modules. The actual
# array conversion still runs through the shared numpy tensor codec.
@pytest.fixture
def fake_torch(monkeypatch):
    import sys
    import types

    import numpy as np

    mod = types.ModuleType("torch")

    class Tensor:
        def __init__(self, arr):
            self._arr = np.asarray(arr)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

        def __dlpack__(self, *args, **kwargs):
            # The real torch.Tensor exports DLPack, so the fake has to as
            # well: without it the generic dlpack rule never matches these
            # values and the tests below would pass with the rule ordering
            # reversed, which is exactly what they exist to catch.
            return self._arr.__dlpack__(*args, **kwargs)

        def __dlpack_device__(self):
            return self._arr.__dlpack_device__()

    mod.Tensor = Tensor
    mod.from_numpy = Tensor
    monkeypatch.setitem(sys.modules, "torch", mod)
    return mod


@pytest.fixture
def fake_jax(monkeypatch):
    import sys
    import types

    import numpy as np

    jax_mod = types.ModuleType("jax")
    jnp_mod = types.ModuleType("jax.numpy")

    class Array:
        def __init__(self, arr):
            self._arr = np.asarray(arr)

        def __array__(self, dtype=None):
            return self._arr if dtype is None else self._arr.astype(dtype)

        def __dlpack__(self, *args, **kwargs):
            # As with the torch fake: real jax.Arrays export DLPack, and
            # omitting it would leave the rule ordering untested.
            return self._arr.__dlpack__(*args, **kwargs)

        def __dlpack_device__(self):
            return self._arr.__dlpack_device__()

    jax_mod.Array = Array
    jnp_mod.asarray = Array
    monkeypatch.setitem(sys.modules, "jax", jax_mod)
    monkeypatch.setitem(sys.modules, "jax.numpy", jnp_mod)
    return jax_mod


class TestTensorLibrarySerialization:
    """torch / jax arrays route through the arrow tensor codec and round-trip
    back to their origin type via the _META_SOURCE tag."""

    def test_detect_torch_tensor_goes_to_arrow_ipc(self, fake_torch):
        import numpy as np

        tensor = fake_torch.Tensor(np.arange(6).reshape(2, 3))

        with tempfile.TemporaryDirectory() as tmpdir:
            result = serialize_value(tensor, Path(tmpdir), "t")

        assert result["content_type"] == ContentType.ARROW_IPC

    def test_roundtrip_torch_tensor(self, fake_torch):
        import numpy as np

        orig = fake_torch.Tensor(np.arange(6, dtype=np.float32).reshape(2, 3))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            meta = serialize_value(orig, tmpdir, "t")
            loaded = deserialize_value(meta["content_type"], tmpdir / meta["file"])

        assert isinstance(loaded, fake_torch.Tensor)
        np.testing.assert_array_equal(loaded.numpy(), orig.numpy())

    def test_torch_tensor_degrades_to_ndarray_when_torch_absent(self, fake_torch):
        import sys

        import numpy as np

        orig = fake_torch.Tensor(np.arange(4.0).reshape(2, 2))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            meta = serialize_value(orig, tmpdir, "t")
            del sys.modules["torch"]  # reader has no torch
            loaded = deserialize_value(meta["content_type"], tmpdir / meta["file"])

        assert isinstance(loaded, np.ndarray)
        np.testing.assert_array_equal(loaded, np.arange(4.0).reshape(2, 2))

    def test_roundtrip_jax_array(self, fake_jax):
        import numpy as np

        orig = fake_jax.Array(np.arange(4, dtype=np.int64).reshape(2, 2))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            meta = serialize_value(orig, tmpdir, "a")
            loaded = deserialize_value(meta["content_type"], tmpdir / meta["file"])

        assert isinstance(loaded, fake_jax.Array)
        np.testing.assert_array_equal(np.asarray(loaded), np.asarray(orig))


class TestArrowTypeRegistry:
    """The detection registry is the single source of truth for arrow routing."""

    def test_unknown_type_still_pickles(self):
        # A set is neither arrow-representable, JSON-safe, nor a module/cell
        # type — it falls through the registry to the pickle catch-all.
        with tempfile.TemporaryDirectory() as tmpdir:
            result = serialize_value({1, 2, 3}, Path(tmpdir), "x")
        assert result["content_type"] == ContentType.PICKLE_OBJECT

    def test_every_rule_predicate_is_callable(self):
        # Guards against registry drift (a rule missing one of its halves).
        for rule in serializer_module._ARROW_TYPE_RULES:
            assert callable(rule.matches)
            assert callable(rule.to_table)


class _CapsuleOnlyTable:
    """A table type pyarrow has never heard of, exporting only the capsule.

    Stands in for duckdb / cudf / ibis / datafusion: the point is that nothing
    in the registry names this class, so if it serializes as Arrow it did so
    through ``_matches_arrow_capsule`` alone.
    """

    def __init__(self, table):
        self._table = table

    def __arrow_c_stream__(self, requested_schema=None):
        return self._table.__arrow_c_stream__(requested_schema)


class _HostileProxy:
    """A stand-in for a detached-session proxy or lazy remote handle.

    Its ``__getattr__`` raises ``RuntimeError``, not ``AttributeError``, which
    is what makes an instance-level ``hasattr`` probe dangerous rather than
    merely wasteful: ``hasattr`` swallows only ``AttributeError``, so anything
    else escapes. That is also what gives the test its teeth — an instance-level
    probe would raise out of the assertion rather than return the wrong answer.
    """

    def __getattr__(self, name):
        raise RuntimeError(f"session detached; cannot load {name!r}")


class _DeviceArrayStub:
    """A dlpack exporter whose buffer lives somewhere numpy cannot read.

    Module level, not nested in the test: the pickle fallback is the assertion,
    and a locally-defined class exercises cloudpickle's by-value path instead
    of the by-reference one a real cell variable would take.
    """

    def __dlpack__(self, *args, **kwargs):
        raise BufferError("cannot export a device buffer to host memory")

    def __dlpack_device__(self):
        return (2, 0)  # kDLCUDA


class _DlpackOnlyArray:
    """An array type exporting only ``__dlpack__``, standing in for cupy / mlx."""

    def __init__(self, arr):
        self._arr = arr

    def __dlpack__(self, *args, **kwargs):
        return self._arr.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self):
        return self._arr.__dlpack_device__()


@contextlib.contextmanager
def _capture_serializer_logs():
    """Collect this module's log records.

    ``caplog`` cannot see them: ``configure_logging`` sets ``propagate = False``
    on the ``strata`` logger, so records never reach the root handler pytest
    installs. Attaching to the module's own logger is the way in.
    """
    records: list[logging.LogRecord] = []
    logger = logging.getLogger("strata.notebook.serializer")
    handler = logging.Handler()
    handler.emit = records.append
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


class TestGenericArrowProtocols:
    """The two hatches that keep unknown libraries out of the pickle path."""

    def test_capsule_only_type_serializes_as_arrow(self):
        value = _CapsuleOnlyTable(pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
        with tempfile.TemporaryDirectory() as tmp:
            meta = serialize_value(value, tmp, "v")
            assert meta["content_type"] == ContentType.ARROW_IPC
            assert meta["rows"] == 3
            back = deserialize_value(meta["content_type"], Path(tmp) / meta["file"])

        # A pa.Table, not pandas: the reader's untagged default is to_pandas(),
        # so this asserts the capsule source tag is both written and honoured.
        assert isinstance(back, pa.Table)
        assert back.column("a").to_pylist() == [1, 2, 3]
        assert back.column("b").to_pylist() == ["x", "y", "z"]

    def test_dlpack_only_type_serializes_as_arrow(self):
        import numpy as np

        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        with tempfile.TemporaryDirectory() as tmp:
            meta = serialize_value(_DlpackOnlyArray(arr), tmp, "v")
            assert meta["content_type"] == ContentType.ARROW_IPC
            back = deserialize_value(meta["content_type"], Path(tmp) / meta["file"])

        assert isinstance(back, np.ndarray)
        assert back.shape == (2, 3, 4)
        assert back.dtype == np.float32
        assert (back == arr).all()

    def test_duckdb_relation_serializes_as_arrow(self):
        # The real-world case the capsule rule exists for, with no shim in the
        # way: a duckdb relation is not picklable, so before this rule the cell
        # that produced one failed outright.
        duckdb = pytest.importorskip("duckdb")

        rel = duckdb.sql("SELECT 1 AS a UNION ALL SELECT 2 ORDER BY a")
        with tempfile.TemporaryDirectory() as tmp:
            meta = serialize_value(rel, tmp, "v")
            assert meta["content_type"] == ContentType.ARROW_IPC
            back = deserialize_value(meta["content_type"], Path(tmp) / meta["file"])

        assert back.column("a").to_pylist() == [1, 2]

    def test_named_rules_still_win_over_the_generic_ones(self):
        # pandas / polars frames export __arrow_c_stream__ and ndarrays export
        # __dlpack__, so the generic rules can only sit at the end of the
        # registry. Reordering them ahead of the named rules downgrades all
        # three of these to pa.Table / ndarray, which is what this pins.
        import numpy as np

        values = {
            "df": pd.DataFrame({"a": [1, 2]}),
            "arr": np.array([1.5, 2.5]),
            # A RecordBatch exports the capsule too, and the generic rule would
            # hand it back as a pa.Table — a quieter identity loss than the
            # DataFrame one, but the same bug.
            "batch": pa.record_batch({"a": [1, 2]}),
        }
        pl = pytest.importorskip("polars")
        values["pdf"] = pl.DataFrame({"a": [1, 2]})
        values["pseries"] = pl.Series("n", [1, 2])

        with tempfile.TemporaryDirectory() as tmp:
            back = {}
            for name, value in values.items():
                meta = serialize_value(value, tmp, name)
                back[name] = deserialize_value(meta["content_type"], Path(tmp) / meta["file"])

        assert isinstance(back["df"], pd.DataFrame)
        assert isinstance(back["arr"], np.ndarray)
        assert isinstance(back["batch"], pa.RecordBatch)
        assert isinstance(back["pdf"], pl.DataFrame)
        assert isinstance(back["pseries"], pl.Series)
        assert back["pseries"].name == "n"

    def test_one_shot_reader_is_left_on_the_pickle_path(self):
        # A RecordBatchReader is consumed by reading. The harness serializes
        # one object twice when a cell's value is also its display output, so
        # routing this through Arrow makes the second artifact report zero
        # rows with no error anywhere. Refusing it keeps the pre-existing
        # visible failure instead of silently storing an empty table.
        table = pa.table({"a": [1, 2, 3]})
        reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches())

        assert hasattr(reader, "__arrow_c_stream__")
        assert serializer_module._matches_arrow_capsule(reader) is False

        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(TypeError):
                serialize_value(reader, tmp, "v")

        # Still readable afterwards: refusing it did not consume the stream.
        assert reader.read_all().num_rows == 3

    def test_class_objects_are_not_mistaken_for_their_instances(self):
        # ``Frame = pd.DataFrame`` binds a class, which carries its instances'
        # protocol methods. Detecting it as tabular sends it down the Arrow
        # path to fail, and the value is not tabular at all.
        import numpy as np

        assert serializer_module._matches_arrow_capsule(pd.DataFrame) is False
        assert serializer_module._matches_dlpack(np.ndarray) is False

        with tempfile.TemporaryDirectory() as tmp:
            meta = serialize_value(pd.DataFrame, tmp, "Frame")
            back = deserialize_value(meta["content_type"], Path(tmp) / meta["file"])

        assert meta["content_type"] == ContentType.PICKLE_OBJECT
        assert back is pd.DataFrame

    def test_hostile_getattr_is_never_invoked_by_detection(self):
        # An instance-level hasattr runs the object's __getattr__. Proxies that
        # raise something other than AttributeError when detached would escape
        # detection entirely — past the Arrow fallback, out of serialize_value —
        # and the caller turns that into an error entry, losing a value that
        # used to pickle without complaint. Probing type(value) never calls it.
        value = _HostileProxy()

        assert serializer_module._matches_arrow_capsule(value) is False
        assert serializer_module._matches_dlpack(value) is False

        with tempfile.TemporaryDirectory() as tmp:
            meta = serialize_value(value, tmp, "proxy")

        assert meta["content_type"] == ContentType.PICKLE_OBJECT

    def test_generic_only_conversion_failure_is_not_warned_about(self):
        # A ChunkedArray exports a non-struct stream, so pa.table() refuses it
        # and pickle is the right answer. The loud "downstream cells expecting
        # tabular shape will break" warning is for a *named* type that failed
        # to encode; claiming it here would be false.
        value = pa.chunked_array([[1, 2], [3]])
        assert serializer_module._matched_only_generic_rules(value) is True

        records = _capture_serializer_logs()
        with records as seen:
            with tempfile.TemporaryDirectory() as tmp:
                meta = serialize_value(value, tmp, "ca")

        assert meta["content_type"] == ContentType.PICKLE_OBJECT
        assert [r.levelno for r in seen if r.levelno >= logging.WARNING] == []

    def test_named_type_conversion_failure_still_warns(self):
        # The counterpart: a structured-dtype ndarray is matched by the *named*
        # numpy rule and then fails to encode, which does break a downstream
        # cell expecting an array. That path keeps its warning.
        import numpy as np

        value = np.array([(1, "x")], dtype=[("i", "i4"), ("s", "U1")])
        assert serializer_module._matched_only_generic_rules(value) is False

        records = _capture_serializer_logs()
        with records as seen:
            with tempfile.TemporaryDirectory() as tmp:
                meta = serialize_value(value, tmp, "sa")

        assert meta["content_type"] == ContentType.PICKLE_OBJECT
        assert [r.levelno for r in seen if r.levelno >= logging.WARNING] != []

    def test_device_buffer_falls_back_to_pickle(self):
        # np.from_dlpack refuses a non-host capsule; the arrow fallback has to
        # turn that into a pickle rather than failing the cell.
        value = _DeviceArrayStub()

        # Asserting the match is what gives this teeth. PICKLE_OBJECT on its
        # own is also what a value the dlpack rule never looked at would
        # produce; the pair says the rule claimed it and the fallback caught it.
        assert serializer_module._matches_dlpack(value) is True

        with tempfile.TemporaryDirectory() as tmp:
            meta = serialize_value(value, tmp, "v")

        assert meta["content_type"] == ContentType.PICKLE_OBJECT


def _arrow_blob(value, name="v"):
    """Serialize *value* the way the harness does and return the raw blob bytes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        meta = serialize_value(value, Path(tmpdir), name)
        return (Path(tmpdir) / meta["file"]).read_bytes()


class TestReadTablePage:
    """read_table_page backs the interactive data viewer's lazy paging."""

    def test_paging_slices_and_reports_total(self):
        df = pd.DataFrame({"a": list(range(100)), "b": [x * 2 for x in range(100)]})
        blob = _arrow_blob(df)

        page = read_table_page(blob, offset=10, limit=5)

        assert page is not None
        assert page["total"] == 100
        assert page["columns"] == ["a", "b"]
        assert page["rows"] == [[10, 20], [11, 22], [12, 24], [13, 26], [14, 28]]

    def test_offset_past_end_returns_empty_page(self):
        blob = _arrow_blob(pd.DataFrame({"a": [1, 2, 3]}))

        page = read_table_page(blob, offset=999, limit=10)

        assert page is not None
        assert page["total"] == 3
        assert page["rows"] == []

    def test_sort_descending_orders_whole_table_before_slicing(self):
        df = pd.DataFrame({"a": [3, 1, 2, 5, 4], "b": ["c", "a", "b", "e", "d"]})
        blob = _arrow_blob(df)

        page = read_table_page(blob, offset=0, limit=2, sort_by="a", sort_dir="desc")

        # Global order is 5,4,3,2,1 — the first page must be the two largest.
        assert page is not None
        assert page["rows"] == [[5, "e"], [4, "d"]]

    def test_unknown_sort_column_is_ignored(self):
        blob = _arrow_blob(pd.DataFrame({"a": [2, 1, 3]}))

        page = read_table_page(blob, sort_by="nope")

        assert page is not None
        assert page["rows"] == [[2], [1], [3]]

    def test_datetime_values_are_json_safe(self):
        df = pd.DataFrame({"t": pd.to_datetime(["2026-01-01", "2026-01-02"])})
        blob = _arrow_blob(df)

        page = read_table_page(blob)

        assert page is not None
        assert all(isinstance(row[0], str) for row in page["rows"])
        json.dumps(page["rows"])  # must not raise

    def test_scalar_shape_is_not_pageable(self):
        # A numpy scalar serializes as arrow shape=scalar — not a table.
        import numpy as np

        page = read_table_page(_arrow_blob(np.int64(7)))

        assert page is None

    def test_non_arrow_blob_returns_none(self):
        assert read_table_page(b"not arrow at all") is None


class TestReadTablePageFiltering:
    """Filter + search narrow the frame before paging; total reflects it."""

    def _blob(self):
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "region": ["North", "South", "north", "East", "West"],
                "revenue": [10.0, 250.0, 99.0, 5.0, 300.0],
            }
        )
        return _arrow_blob(df)

    def test_global_search_is_case_insensitive_across_columns(self):
        page = read_table_page(self._blob(), search="north")

        assert page["total"] == 2  # "North" and "north"
        assert {row[1] for row in page["rows"]} == {"North", "north"}

    def test_search_matches_numeric_columns_cast_to_string(self):
        page = read_table_page(self._blob(), search="250")

        assert page["total"] == 1
        assert page["rows"][0][0] == 2

    def test_filter_greater_than_on_numeric_column(self):
        page = read_table_page(self._blob(), filters=[{"col": "revenue", "op": "gt", "value": 100}])

        assert page["total"] == 2
        assert sorted(row[0] for row in page["rows"]) == [2, 5]

    def test_filter_between_inclusive(self):
        page = read_table_page(
            self._blob(),
            filters=[{"col": "revenue", "op": "between", "value": 10, "value2": 99}],
        )

        assert sorted(row[0] for row in page["rows"]) == [1, 3]

    def test_filter_contains_on_string_column(self):
        page = read_table_page(
            self._blob(), filters=[{"col": "region", "op": "contains", "value": "th"}]
        )

        # "North", "South", "north" all contain "th" (case-insensitive)
        assert page["total"] == 3

    def test_filters_and_search_compose(self):
        page = read_table_page(
            self._blob(),
            filters=[{"col": "revenue", "op": "gt", "value": 50}],
            search="north",
        )

        assert page["total"] == 1
        assert page["rows"][0][0] == 3  # "north", revenue 99

    def test_uncoercible_filter_value_is_skipped(self):
        # "abc" can't coerce to the int column — filter is a no-op, not a 500.
        page = read_table_page(self._blob(), filters=[{"col": "id", "op": "eq", "value": "abc"}])

        assert page["total"] == 5

    def test_unknown_column_filter_is_skipped(self):
        page = read_table_page(self._blob(), filters=[{"col": "nope", "op": "eq", "value": 1}])

        assert page["total"] == 5


class TestReadTableSummary:
    def test_summary_reports_dtype_nulls_distinct_and_extent(self):
        from strata.notebook.serializer import read_table_summary

        df = pd.DataFrame(
            {
                "n": [1, 2, 2, None],
                "label": ["a", "b", "a", "c"],
            }
        )
        summary = read_table_summary(_arrow_blob(df))

        assert summary["total"] == 4
        by_name = {c["name"]: c for c in summary["columns"]}
        assert by_name["n"]["nulls"] == 1
        assert by_name["n"]["distinct"] == 2  # {1, 2}
        assert by_name["n"]["min"] == 1
        assert by_name["n"]["max"] == 2
        # String columns get null/distinct but no min/max.
        assert by_name["label"]["distinct"] == 3
        assert by_name["label"]["min"] is None

    def test_summary_none_for_non_table(self):
        import numpy as np

        from strata.notebook.serializer import read_table_summary

        assert read_table_summary(_arrow_blob(np.int64(3))) is None


class TestWriteTableExport:
    def test_csv_export_roundtrips_and_respects_filter(self):
        import io

        from strata.notebook.serializer import write_table_export

        df = pd.DataFrame({"id": [1, 2, 3], "v": [10, 200, 30]})
        blob = _arrow_blob(df)

        raw = write_table_export(blob, "csv", filters=[{"col": "v", "op": "gt", "value": 50}])
        out = pd.read_csv(io.BytesIO(raw))

        assert list(out.columns) == ["id", "v"]
        assert out["id"].tolist() == [2]

    def test_parquet_export_roundtrips(self):
        import io

        from strata.notebook.serializer import write_table_export

        df = pd.DataFrame({"id": [1, 2], "v": [10, 20]})
        raw = write_table_export(_arrow_blob(df), "parquet")
        out = pd.read_parquet(io.BytesIO(raw))

        assert out["id"].tolist() == [1, 2]

    def test_unknown_format_returns_none(self):
        from strata.notebook.serializer import write_table_export

        assert write_table_export(_arrow_blob(pd.DataFrame({"a": [1]})), "xlsx") is None
