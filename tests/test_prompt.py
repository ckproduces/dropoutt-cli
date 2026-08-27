from dropoutt.prompt import next_index


def test_next_index_wraps():
    assert next_index(0, 2, "up") == 1
    assert next_index(1, 2, "down") == 0
    assert next_index(0, 2, "down") == 1
    assert next_index(1, 2, "up") == 0
    assert next_index(0, 2, "enter") == 0
    assert next_index(0, 0, "down") == 0
