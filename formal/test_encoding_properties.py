"""Property tests: provenance encodings and the names a table answers to.

PROV-3: a transform spec's canonical JSON reads back as the spec it came from, with
every value's type (``1``, ``1.0`` and ``true`` stay apart), so different specs never
share bytes; and the bytes do not depend on the order ``params`` was built in.

ACL-2: a deny rule written for any name of a table refuses every URI that can load it.
The oracle says which URIs load one table without Strata's naming code: under a SQL
catalog every warehouse URI reads one catalog, and a bare name joins them when the
default catalog is ``strata``; a table may answer to any spelling of its name (Glue and
Hive resolve names case-insensitively). A named catalog over the same data is not
detectable and is what the documented ``*:`` prefix is for, so it is its own class.

    uv run --with hypothesis pytest formal/test_encoding_properties.py
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
from vestigo.provenance import TransformSpec

from strata.config import AclConfig, AclRule, StrataConfig
from strata.types import Principal

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

EXAMPLES = int(os.environ.get("ENCODING_EXAMPLES", "500"))

_SCALARS = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**70), max_value=2**70)
    | st.sampled_from([0, 1, -1, 0.0, -0.0, 1.0, -1.0, True, False])
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=8)
)
_JSON = st.recursive(
    _SCALARS,
    lambda inner: (
        st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=6), inner, max_size=4)
    ),
    max_leaves=20,
)


def _same(a: object, b: object) -> bool:
    """Equal with types: ``1``, ``1.0`` and ``True`` differ, as do ``0.0`` and ``-0.0``."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)  # type: ignore[union-attr]
    if isinstance(a, list):
        return len(a) == len(b) and all(map(_same, a, b))  # type: ignore[arg-type]
    if isinstance(a, float):
        return a.hex() == b.hex()  # type: ignore[union-attr]
    return a == b


def _reversed_order(value: object) -> object:
    if isinstance(value, dict):
        return {k: _reversed_order(value[k]) for k in reversed(list(value))}
    if isinstance(value, list):
        return [_reversed_order(item) for item in value]
    return value


@settings(max_examples=EXAMPLES, deadline=None)
@given(
    executor=st.text(max_size=12),
    params=st.dictionaries(st.text(max_size=6), _JSON, max_size=5),
    inputs=st.lists(st.text(max_size=10), max_size=4),
)
def test_a_transform_spec_reads_back_as_itself(executor, params, inputs):
    spec = TransformSpec(executor=executor, params=params, inputs=inputs)

    read = json.loads(spec.canonical_json)

    assert _same(read, {"executor": executor, "params": params, "inputs": inputs})
    rebuilt = TransformSpec(executor=executor, params=_reversed_order(params), inputs=inputs)
    assert rebuilt.canonical_json == spec.canonical_json


# --- ACL-2 -------------------------------------------------------------------

_WAREHOUSES = {
    "s3": "s3://bucket/wh#",
    "gs": "gs://bucket/wh#",
    "az": "abfss://box@acct.dfs.core.windows.net/wh#",
    "file": "/no/such/wh#",
    "file-uri": "file:///no/such/wh#",
}
_ACL_PREFIX = {"s3": "s3", "gs": "gs", "az": "az", "file": "file", "file-uri": "file"}


def _spellings(name: str) -> st.SearchStrategy[str]:
    return st.lists(st.booleans(), min_size=len(name), max_size=len(name)).map(
        lambda upper: "".join(
            c.upper() if u else c.lower() for c, u in zip(name, upper, strict=True)
        )
    )


def _forms(shape: str, catalog_name: str) -> dict[str, tuple[str, str, str]]:
    """Each address form: (class, how the URI starts, the ACL prefix the oracle gives it)."""
    shared = shape == "sql"
    forms = {
        form: ("warehouse" if shared else form, start, _ACL_PREFIX[form])
        for form, start in _WAREHOUSES.items()
    }
    forms["bare"] = (
        "warehouse" if shared and catalog_name == "strata" else "bare",
        "",
        "file",
    )
    forms["named"] = ("lake", "lake:", "lake")
    return forms


@st.composite
def _case(draw):
    shape = draw(st.sampled_from(["sql", "own"]))
    catalog_name = draw(st.sampled_from(["strata", "other"]))
    forms = _forms(shape, catalog_name)
    denied = draw(st.sampled_from(sorted(forms)))
    namespace, table = draw(st.sampled_from([("fin", "ledger"), ("db", "events")]))
    deny_name = draw(_spellings(f"{namespace}.{table}"))
    requests = [
        (form, draw(_spellings(f"{namespace}.{table}")))
        for form in sorted(forms)
        if forms[form][0] == forms[denied][0]
    ]
    return shape, catalog_name, forms, denied, deny_name, requests


@settings(
    max_examples=EXAMPLES,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(case=_case())
def test_a_deny_on_any_name_refuses_every_uri_of_the_table(case, tmp_path, monkeypatch):
    from fastapi import HTTPException

    import strata.server as server_module
    from strata.api.dependencies import authorize_table_access
    from strata.auth.principal import set_principal
    from strata.iceberg.catalog import table_identity_for

    shape, catalog_name, forms, denied, deny_name, requests = case
    _, _, prefix = forms[denied]
    config = StrataConfig(
        deployment_mode="service",
        artifact_dir=tmp_path / "artifacts",
        cache_dir=tmp_path / "cache",
        auth_mode="trusted_proxy",
        proxy_token="t",
        catalog_name=catalog_name,
        catalog_properties={"uri": "sqlite:///unused.db"} if shape == "sql" else {},
        catalogs={"lake": {"type": "sql", "uri": "sqlite:///lake.db"}},
        acl_config=AclConfig(
            default="allow", deny_rules=[AclRule(principal="*", tables=(f"{prefix}:{deny_name}",))]
        ),
    )
    monkeypatch.setattr(server_module, "get_state", lambda: SimpleNamespace(config=config))
    set_principal(Principal(id="alice"))
    try:
        for form, spelling in requests:
            uri = forms[form][1] + spelling
            with pytest.raises(HTTPException):
                authorize_table_access(uri, table_identity_for(uri, config))
    finally:
        set_principal(None)
