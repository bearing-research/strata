# Unit tests for the "Add computed columns" cell.
#
# This cell consumes ``sales`` from upstream and adds ``revenue`` and ``month``.
# The ``cell`` fixture re-runs the cell against its real upstream input, so
# ``cell.sales`` here is the post-transform frame.


def test_revenue_is_units_times_price(cell):
    expected = cell.sales["units"] * cell.sales["price"]
    assert (cell.sales["revenue"] == expected).all()


def test_revenue_is_positive(cell):
    assert (cell.sales["revenue"] > 0).all()


def test_month_is_a_year_month_string(cell):
    assert cell.sales["month"].str.fullmatch(r"\d{4}-\d{2}").all()
