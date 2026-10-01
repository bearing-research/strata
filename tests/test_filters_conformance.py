"""Conformance tests for the duplicated filter modules.

``src/strata/filters.py`` and ``packages/strata-client/src/strata_client/filters.py`` share only the
JSON wire format by design; these pin fingerprints and value serialization across both.
"""

import uuid
from datetime import date, datetime
from datetime import time as time_of_day
from decimal import Decimal
from pathlib import Path

import pytest
import strata_client.filters as client_filters

import strata.filters as server_filters

BOTH = pytest.mark.parametrize("mod", [server_filters, client_filters], ids=["server", "client"])

_VALUES = [
    1,
    -3,
    "x",
    "1",  # string '1' must stay distinct from int 1
    True,  # bool must stay distinct from int 1
    3.5,
    datetime(2020, 1, 2, 3, 4, 5),
    date(2020, 1, 2),
    time_of_day(3, 4, 5),
    Decimal("1.50"),
    uuid.UUID("12345678-1234-5678-1234-567812345678"),
    b"\x00abc\xff",
]


def test_source_bodies_are_identical():
    """The two filter modules must stay byte-identical below the docstring."""

    def _code(path: str) -> str:
        text = Path(path).read_text()
        # Drop the module docstring; the import block onward must match exactly.
        return text[text.index("import base64") :]

    assert _code("src/strata/filters.py") == _code(
        "packages/strata-client/src/strata_client/filters.py"
    )


@BOTH
@pytest.mark.parametrize("value", _VALUES)
def test_value_round_trips(mod, value):
    assert mod.deserialize_filter_value(mod.serialize_filter_value(value)) == value


@BOTH
def test_serialized_values_are_json_native(mod):
    import json

    for value in _VALUES:
        json.dumps(mod.serialize_filter_value(value))  # must not raise


@BOTH
def test_unsupported_value_type_rejected(mod):
    with pytest.raises(TypeError):
        mod.serialize_filter_value(object())


@BOTH
def test_fingerprint_no_field_boundary_collision(mod):
    """('a>','=') and ('a','>=') must not concatenate to the same string."""
    fp1 = mod.compute_filter_fingerprint([mod.Filter("a>", mod.FilterOp.EQ, 1)])
    fp2 = mod.compute_filter_fingerprint([mod.Filter("a", mod.FilterOp.GE, 1)])
    assert fp1 != fp2


@BOTH
def test_fingerprint_distinguishes_value_type(mod):
    fp_int = mod.compute_filter_fingerprint([mod.Filter("c", mod.FilterOp.EQ, 1)])
    fp_str = mod.compute_filter_fingerprint([mod.Filter("c", mod.FilterOp.EQ, "1")])
    assert fp_int != fp_str


@BOTH
def test_fingerprint_order_independent(mod):
    f1 = mod.Filter("x", mod.FilterOp.EQ, 1)
    f2 = mod.Filter("y", mod.FilterOp.LT, 2)
    assert mod.compute_filter_fingerprint([f1, f2]) == mod.compute_filter_fingerprint([f2, f1])


@BOTH
def test_fingerprint_empty_is_nofilter(mod):
    assert mod.compute_filter_fingerprint(None) == "nofilter"
    assert mod.compute_filter_fingerprint([]) == "nofilter"


def test_both_copies_agree_on_fingerprint():
    """Same filters, same fingerprint from either copy."""
    sf = [server_filters.Filter("ts", server_filters.FilterOp.GE, datetime(2021, 5, 1))]
    cf = [client_filters.Filter("ts", client_filters.FilterOp.GE, datetime(2021, 5, 1))]
    assert server_filters.compute_filter_fingerprint(
        sf
    ) == client_filters.compute_filter_fingerprint(cf)


class TestFilterSpecOpValidation:
    """An invalid operator fails validation (400), not later as an uncaught ValueError."""

    def test_invalid_op_rejected_at_validation(self):
        from pydantic import ValidationError

        from strata.types import IdentityParams

        with pytest.raises(ValidationError):
            IdentityParams.model_validate({"filters": [{"column": "a", "op": "LIKE", "value": 1}]})

    def test_valid_op_converts(self):
        from strata.types import IdentityParams

        params = IdentityParams.model_validate(
            {"filters": [{"column": "a", "op": ">=", "value": 1}]}
        )
        filters = params.to_strata_filters()
        assert filters[0].op.value == ">="


class TestAdapterToServerFilterRoundTrip:
    """A richer-typed filter value survives adapter, JSON wire and ``FilterSpec`` intact.

    The planner compares it against Parquet column stats, so the Python type must survive.
    """

    _ADAPTERS = ["pandas", "arrow", "polars", "duckdb", "datafusion"]
    _NON_PRIMITIVE = [
        datetime(2021, 5, 1, 12, 30, 5),
        date(2021, 5, 1),
        time_of_day(12, 30, 5),
        Decimal("1.50"),
        uuid.UUID("12345678-1234-5678-1234-567812345678"),
        b"\x00abc\xff",
    ]

    @pytest.mark.parametrize("adapter_name", _ADAPTERS)
    @pytest.mark.parametrize("value", _NON_PRIMITIVE)
    def test_value_survives_adapter_to_server(self, adapter_name, value):
        import importlib
        import json

        from strata_client.filters import Filter as ClientFilter
        from strata_client.filters import FilterOp as ClientFilterOp

        from strata.types import IdentityParams

        adapter = importlib.import_module(f"strata_client.integration.{adapter_name}")
        transform = adapter._build_scan_transform(
            filters=[ClientFilter("col", ClientFilterOp.EQ, value)]
        )
        # Simulate the JSON wire hop of the client-to-server request.
        params = json.loads(json.dumps(transform["params"]))
        identity = IdentityParams.model_validate(params)
        decoded = identity.to_strata_filters()[0].value

        assert decoded == value
        # datetime is a subclass of date, so assert the exact reconstructed type.
        assert type(decoded) is type(value)
