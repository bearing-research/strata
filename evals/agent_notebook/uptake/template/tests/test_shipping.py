from harbor.shipping import eta_days


def test_a_region_to_itself_is_a_day():
    assert eta_days("central", "central") == 1
