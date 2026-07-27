"""P2.5 tests: the access manifest (experiment<->production parity contract).

Covers declared access (mounts / tables / connections) captured reliably plus
best-effort parity-risk detection (unpinned tables, rw / local mounts, local
file reads, external registries, SQL engine mismatch).
"""

from __future__ import annotations

from pathlib import Path

from strata.notebook.compile import build_pipeline_ir_from_dir
from strata.pipeline import build_access_manifest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _manifest(example: str, *, runtime: str = "container"):
    return build_access_manifest(build_pipeline_ir_from_dir(EXAMPLES / example, runtime=runtime))


def _write_nb(root: Path, cells: list[tuple[str, str]], *, extra_toml: str = "") -> Path:
    """Scaffold a minimal notebook dir; cells = [(id, source), ...]."""
    (root / "cells").mkdir(parents=True)
    lines = ['notebook_id = "acc-001"', 'name = "Access"', extra_toml, "cells = ["]
    for i, (cid, _src) in enumerate(cells):
        lines.append(
            f'    {{ id = "{cid}", file = "{cid}.py", language = "python", order = {i} }},'
        )
    lines.append("]")
    (root / "notebook.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for cid, src in cells:
        (root / "cells" / f"{cid}.py").write_text(src, encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# Declared access
# ---------------------------------------------------------------------------


def test_s3_mount_captured_without_secret_values():
    m = _manifest("s3_mount")
    assert len(m.mounts) == 1
    mount = m.mounts[0]
    assert mount.name == "jfk_weather"
    assert mount.scheme == "s3" and mount.mode == "ro"
    # Only option KEYS, never values (they can be credentials).
    assert mount.option_keys == ["anon"]
    # A read-only s3 mount is not itself a parity risk.
    assert m.parity_risks == []


def test_sql_connection_captured():
    m = _manifest("sql_orders_report", runtime="glue")
    names = {c.name for c in m.connections}
    assert "warehouse" in names
    conn = next(c for c in m.connections if c.name == "warehouse")
    assert conn.driver == "sqlite"


def test_iris_has_no_external_access():
    m = _manifest("iris_classification")
    assert m.mounts == [] and m.tables == [] and m.connections == []


# ---------------------------------------------------------------------------
# Parity risks
# ---------------------------------------------------------------------------


def test_local_database_connection_is_flagged():
    # sql_orders_report uses a local sqlite file — unreachable from Glue.
    m = _manifest("sql_orders_report", runtime="glue")
    assert any("local sqlite" in r and "warehouse" in r for r in m.parity_risks)


def test_local_file_read_is_flagged(tmp_path):
    nb = _write_nb(
        tmp_path / "nb",
        [("load", "import pandas as pd\ndf = pd.read_csv('/data/train.csv')\n")],
    )
    m = build_access_manifest(build_pipeline_ir_from_dir(nb))
    assert any("/data/train.csv" in r and "local" in r for r in m.parity_risks)


def test_s3_uri_read_is_not_flagged_as_local(tmp_path):
    nb = _write_nb(
        tmp_path / "nb",
        [("load", "import pandas as pd\ndf = pd.read_parquet('s3://bucket/train.parquet')\n")],
    )
    m = build_access_manifest(build_pipeline_ir_from_dir(nb))
    assert not any("local" in r for r in m.parity_risks)


def test_external_registry_import_is_flagged(tmp_path):
    nb = _write_nb(
        tmp_path / "nb",
        [("m", "import mlflow\nmodel = mlflow.sklearn.load_model('models:/m/1')\n")],
    )
    m = build_access_manifest(build_pipeline_ir_from_dir(nb))
    assert any("MLflow" in r for r in m.parity_risks)


def test_unpinned_table_flagged_pinned_is_not(tmp_path):
    unpinned = _write_nb(
        tmp_path / "u",
        [("scan", "# @table events warehouse#db.events\nrows = len(events)\n")],
    )
    mu = build_access_manifest(build_pipeline_ir_from_dir(unpinned))
    assert mu.tables and mu.tables[0].name == "events"
    assert any("not snapshot-pinned" in r for r in mu.parity_risks)

    pinned = _write_nb(
        tmp_path / "p",
        [("scan", "# @table events warehouse#db.events snapshot=42\nrows = len(events)\n")],
    )
    mp = build_access_manifest(build_pipeline_ir_from_dir(pinned))
    assert mp.tables and mp.tables[0].snapshot_pin == 42
    assert not any("not snapshot-pinned" in r for r in mp.parity_risks)
