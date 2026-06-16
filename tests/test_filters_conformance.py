"""Conformance tests for the duplicated filter modules.

``src/strata/filters.py`` and ``packages/strata-client/src/strata_client/filters.py``
are duplicated *by design* (server and client share no code, only the JSON wire
format). These tests pin the behavior that must stay identical across both
copies — fingerprint stability/collision-safety and value serialization — and
guard against the two files drifting.
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
    """The bug: ('a>','=') and ('a','>=') concatenated to the same string."""
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
    """Same filters → same fingerprint from either copy (shared wire format)."""
    sf = [server_filters.Filter("ts", server_filters.FilterOp.GE, datetime(2021, 5, 1))]
    cf = [client_filters.Filter("ts", client_filters.FilterOp.GE, datetime(2021, 5, 1))]
    assert server_filters.compute_filter_fingerprint(
        sf
    ) == client_filters.compute_filter_fingerprint(cf)


class TestFilterSpecOpValidation:
    """Finding #3: an invalid operator must fail validation (→ 400), not escape
    as an uncaught ValueError from FilterOp(f.op) later."""

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
