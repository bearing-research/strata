# Unit tests for the "Create sample sales dataset" cell.
#
# The ``cell`` fixture exposes the cell's namespace after it runs: ``cell.X`` is
# whatever ``X`` is at the end of the cell — here, the ``sales`` DataFrame the
# cell builds. Run these from the Tests panel on the cell (or via the WS
# ``cell_run_tests`` request); pytest collects every ``test_*`` function below.


def test_row_count(cell):
    assert len(cell.sales) == 200


def test_expected_columns(cell):
    assert list(cell.sales.columns) == ["date", "region", "product", "units", "price"]


def test_no_missing_values(cell):
    assert cell.sales.isnull().sum().sum() == 0


def test_values_are_in_range(cell):
    assert cell.sales["units"].between(1, 49).all()
    assert cell.sales["price"].between(5, 100).all()
    assert set(cell.sales["region"]) <= {"North", "South", "East", "West"}
