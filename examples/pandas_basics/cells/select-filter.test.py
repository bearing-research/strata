# Unit tests for the "Filter high-value orders" cell.
#
# ``high_value`` keeps only orders with units > 20 AND price > 30. Tests pin
# that invariant — they would fail loudly if the filter were ever loosened.


def test_predicate_holds_for_every_row(cell):
    assert (cell.high_value["units"] > 20).all()
    assert (cell.high_value["price"] > 30).all()


def test_high_value_is_a_subset_of_sales(cell):
    assert len(cell.high_value) <= len(cell.sales)
