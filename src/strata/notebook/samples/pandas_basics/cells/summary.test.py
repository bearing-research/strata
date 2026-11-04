# Unit tests for the "Top product per region" cell.
#
# ``top_products`` should name exactly one winning product per region.


def test_one_row_per_region(cell):
    assert cell.top_products["region"].is_unique


def test_covers_every_region(cell):
    assert set(cell.top_products["region"]) == {"North", "South", "East", "West"}
