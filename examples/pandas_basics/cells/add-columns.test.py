# Unit tests for the "Add computed columns" cell.
#
# This cell reads ``sales`` from upstream and adds ``revenue`` and ``month`` in a
# new frame, ``orders``. The ``cell`` fixture re-runs the cell against its real
# upstream input.


def test_revenue_is_units_times_price(cell):
    expected = cell.orders["units"] * cell.orders["price"]
    assert (cell.orders["revenue"] == expected).all()


def test_revenue_is_positive(cell):
    assert (cell.orders["revenue"] > 0).all()


def test_month_is_a_year_month_string(cell):
    assert cell.orders["month"].str.fullmatch(r"\d{4}-\d{2}").all()
