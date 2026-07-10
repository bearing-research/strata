"""Tests for serializer module."""

from __future__ import annotations

import json
import pickle
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


def _mark_as_cell_module(cls, module_source: str) -> None:
    import sys

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
