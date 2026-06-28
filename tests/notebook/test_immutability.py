"""Tests for immutability detection (M6)."""

from __future__ import annotations

import pandas as pd
import pytest

from strata.notebook.immutability import (
    InputSnapshot,
    apply_defensive_copy,
    detect_mutations,
    snapshot_inputs,
)


class TestMutationDetection:
    """Test mutation detection for various types."""

    def test_dataframe_mutation_detection(self):
        """Test detecting DataFrame mutation via inplace operation."""
        # Create a DataFrame
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

        # Create namespace with the DataFrame
        namespace = {"df": df}

        # Take snapshot
        snapshots = snapshot_inputs(namespace, ["df"])
        assert len(snapshots) == 1
        assert snapshots[0].var_name == "df"
        assert snapshots[0].content_hash is not None

        # Mutate the DataFrame
        df.drop("a", axis=1, inplace=True)

        # Detect mutations
        warnings = detect_mutations(namespace, snapshots)

        # Should detect the mutation
        assert len(warnings) > 0
        assert "mutated" in warnings[0]["message"].lower()

    def test_no_mutation_on_reassignment(self):
        """Test that reassignment is not detected as mutation."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        namespace = {"df": df}

        # Take snapshot
        snapshots = snapshot_inputs(namespace, ["df"])
        original_id = snapshots[0].identity

        # Reassign the variable
        namespace["df"] = df.drop("a", axis=1)

        # The new DataFrame has a different id
        assert id(namespace["df"]) != original_id

        # Should NOT detect mutation (identity changed)
        warnings = detect_mutations(namespace, snapshots)
        assert len(warnings) == 0

    def test_no_mutation_on_read_only_access(self):
        """Test that read-only operations don't trigger warnings."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        namespace = {"df": df}

        # Take snapshot
        snapshots = snapshot_inputs(namespace, ["df"])

        # Read-only operations (these don't mutate)
        _ = df.describe()
        _ = df.shape
        _ = df.loc[0]

        # Should NOT detect mutations
        warnings = detect_mutations(namespace, snapshots)
        assert len(warnings) == 0

    def test_snapshot_multiple_inputs(self):
        """Test snapshotting multiple input variables."""
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"y": [3, 4]})
        scalar = 42

        namespace = {"df1": df1, "df2": df2, "scalar": scalar}

        # Take snapshot of all inputs
        snapshots = snapshot_inputs(namespace, ["df1", "df2", "scalar"])

        assert len(snapshots) == 3
        assert snapshots[0].var_name == "df1"
        assert snapshots[1].var_name == "df2"
        assert snapshots[2].var_name == "scalar"

    def test_deleted_input_detected(self):
        """Test that deleted input is detected as mutation."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        namespace = {"df": df}

        # Take snapshot
        snapshots = snapshot_inputs(namespace, ["df"])

        # Delete the variable
        del namespace["df"]

        # detect_mutations flags deleted variables as mutations
        warnings = detect_mutations(namespace, snapshots)
        assert isinstance(warnings, list)
        assert len(warnings) == 1, "Expected one warning for deleted variable"
        assert "df" in warnings[0]["var_name"]

    def test_defensive_copy_arrow(self):
        """Test defensive copy for arrow content type."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        # Arrow IPC doesn't need a copy (deserialization produces new object)
        copy_df = apply_defensive_copy(df, "arrow/ipc")

        # For arrow, we return the same object (no copy needed)
        assert copy_df is df

    def test_defensive_copy_json(self):
        """Test defensive copy for JSON content type."""
        data = {"key": "value", "list": [1, 2, 3]}

        # JSON objects get shallow copy
        copy_data = apply_defensive_copy(data, "json/object")

        # Should be a different object
        assert copy_data is not data

        # But shallow copy means nested objects are shared
        assert copy_data == data
        assert copy_data["list"] is data["list"]

    def test_defensive_copy_pickle(self):
        """Test defensive copy for pickle content type."""

        class CustomClass:
            def __init__(self, value):
                self.value = value

        obj = CustomClass(42)

        # Pickle objects get deep copy
        copy_obj = apply_defensive_copy(obj, "pickle/object")

        # Should be a different object
        assert copy_obj is not obj

        # But values should be equal
        assert copy_obj.value == obj.value


class TestInputSnapshot:
    """Test InputSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test creating an InputSnapshot."""
        snapshot = InputSnapshot(
            var_name="test_var",
            identity=12345,
            content_hash="abc123",
        )

        assert snapshot.var_name == "test_var"
        assert snapshot.identity == 12345
        assert snapshot.content_hash == "abc123"

    def test_snapshot_with_none_hash(self):
        """Test snapshot with None content hash."""
        snapshot = InputSnapshot(
            var_name="test_var",
            identity=12345,
            content_hash=None,
        )

        assert snapshot.content_hash is None


class TestFingerprintRegistry:
    """Runtime detection now covers numpy / dicts / lists / sized containers,
    not just pandas (design-mutation-fingerprint-registry)."""

    def _warned(self, value, mutate) -> bool:
        namespace = {"v": value}
        snapshots = snapshot_inputs(namespace, ["v"])
        mutate(namespace["v"])
        return len(detect_mutations(namespace, snapshots)) > 0

    def test_numpy_inplace_mutation_detected(self):
        import numpy as np

        assert self._warned(np.array([1, 2, 3, 4]), lambda a: a.fill(9))

    def test_numpy_no_mutation_not_flagged(self):
        import numpy as np

        assert not self._warned(np.array([1.0, 2.0, 3.0]), lambda a: a.sum())

    def test_dict_key_added_detected(self):
        assert self._warned({"a": 1}, lambda d: d.update({"z": 9}))

    def test_dict_no_mutation_not_flagged(self):
        assert not self._warned({"a": 1, "b": 2}, lambda d: d.get("a"))

    def test_list_append_detected(self):
        assert self._warned([1, 2, 3], lambda lst: lst.append(4))

    def test_list_sort_detected(self):
        assert self._warned([3, 1, 2], lambda lst: lst.sort())

    def test_list_no_mutation_not_flagged(self):
        assert not self._warned([1, 2, 3], lambda lst: lst.index(2))

    def test_set_add_detected_via_sized_fallback(self):
        assert self._warned({1, 2, 3}, lambda s: s.add(9))

    def test_opaque_object_mutation_detected_via_general_fingerprint(self):
        """An arbitrary picklable object is content-checked via the general
        serializer fallback — no per-type rule needed (the Phase 2a fix)."""

        class Opaque:
            def __init__(self):
                self.x = 1

        namespace = {"v": Opaque()}
        snapshots = snapshot_inputs(namespace, ["v"])
        assert snapshots[0].content_hash is not None  # general fingerprint applies
        namespace["v"].x = 2
        assert len(detect_mutations(namespace, snapshots)) == 1

    def test_opaque_object_no_mutation_not_flagged(self):
        class Opaque:
            def __init__(self):
                self.x = 1

        namespace = {"v": Opaque()}
        snapshots = snapshot_inputs(namespace, ["v"])
        _ = namespace["v"].x  # read only, no mutation
        assert detect_mutations(namespace, snapshots) == []

    def test_unpicklable_object_is_identity_only_no_crash(self):
        """A value the serializer can't handle falls back to identity-only."""
        namespace = {"v": lambda: 1}  # lambdas aren't cloudpickle-stable here
        snapshots = snapshot_inputs(namespace, ["v"])
        # Either fingerprinted or not, mutation detection must not raise.
        assert detect_mutations(namespace, snapshots) == []

    def test_string_input_is_not_fingerprinted(self):
        """str is immutable — skipped by the general fallback, identity-only."""
        namespace = {"v": "hello"}
        snapshots = snapshot_inputs(namespace, ["v"])
        assert snapshots[0].content_hash is None

    def test_exported_mutation_is_not_warned(self):
        """A mutated input the cell also exported reaches downstream correctly —
        only the *unexported* mutation (silent-stale) warns."""

        class Opaque:
            def __init__(self):
                self.x = 1

        namespace = {"v": Opaque()}
        snapshots = snapshot_inputs(namespace, ["v"])
        namespace["v"].x = 2
        # Not exported → warns.
        assert len(detect_mutations(namespace, snapshots)) == 1
        # Exported (re-captured) → silent.
        assert detect_mutations(namespace, snapshots, exported_names={"v"}) == []


@pytest.fixture
def fake_torch(monkeypatch):
    """API-compatible torch stub (real torch is too heavy for CI). The
    fingerprint runs through the same numpy-backed sample path as real torch."""
    import sys
    import types

    import numpy as np

    mod = types.ModuleType("torch")

    class Tensor:
        def __init__(self, arr, device="cpu"):
            self._arr = np.asarray(arr)
            self.device = device

        @property
        def shape(self):
            return self._arr.shape

        @property
        def dtype(self):
            return self._arr.dtype

        def detach(self):
            return self

        def flatten(self):
            return Tensor(self._arr.reshape(-1), self.device)

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

        def __getitem__(self, key):
            return Tensor(self._arr[key], self.device)

        def __len__(self):
            return len(self._arr)

        def zero_(self):
            self._arr[:] = 0
            return self

        def add_(self, value):
            self._arr += value
            return self

    mod.Tensor = Tensor
    monkeypatch.setitem(sys.modules, "torch", mod)
    return mod


class TestTorchFingerprint:
    """torch tensors get element-level detection via sys.modules probing."""

    def _warned(self, value, mutate) -> bool:
        namespace = {"v": value}
        snapshots = snapshot_inputs(namespace, ["v"])
        mutate(namespace["v"])
        return len(detect_mutations(namespace, snapshots)) > 0

    def test_torch_inplace_zero_detected(self, fake_torch):
        assert self._warned(fake_torch.Tensor([1.0, 2.0, 3.0, 4.0]), lambda t: t.zero_())

    def test_torch_inplace_add_detected(self, fake_torch):
        assert self._warned(fake_torch.Tensor([1.0, 2.0, 3.0]), lambda t: t.add_(5))

    def test_torch_no_mutation_not_flagged(self, fake_torch):
        assert not self._warned(fake_torch.Tensor([1.0, 2.0, 3.0]), lambda t: t.numpy().sum())

    def test_torch_detected_without_importing_torch(self, fake_torch):
        # The rule probes sys.modules — a tensor implies torch is already there.
        import sys

        snapshots = snapshot_inputs({"v": fake_torch.Tensor([1, 2, 3])}, ["v"])
        assert snapshots[0].content_hash is not None
        assert isinstance(sys.modules["torch"], type(fake_torch))
