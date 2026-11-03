import pytest

from linked_list import LinkedList


def snapshot(ll):
    return ll.to_list(), len(ll)


def build(values):
    ll = LinkedList()
    for v in values:
        ll.insert(v)
    return ll


# Criterion 1
def test_append_1_2_3():
    ll = LinkedList()
    ll.insert(1)
    ll.insert(2)
    ll.insert(3)
    assert ll.to_list() == [1, 2, 3]
    assert len(ll) == 3


def test_append_with_explicit_none_index():
    ll = build([1, 2])
    ll.insert(3, None)
    assert list(ll) == [1, 2, 3]
    assert len(ll) == 3


# Criterion 2
def test_insert_at_index_zero():
    ll = build([1, 2])
    ll.insert(0, 0)
    assert ll.to_list() == [0, 1, 2]
    assert len(ll) == 3


def test_insert_middle():
    ll = build([1, 3])
    ll.insert(2, 1)
    assert ll.to_list() == [1, 2, 3]
    assert len(ll) == 3


def test_insert_at_len():
    ll = build([1, 2])
    ll.insert(3, len(ll))
    assert ll.to_list() == [1, 2, 3]
    assert len(ll) == 3


def test_insert_multiple_positions_sequence():
    ll = build([10, 30])
    ll.insert(20, 1)
    ll.insert(0, 0)
    ll.insert(40, len(ll))
    ll.insert(25, 3)
    assert ll.to_list() == [0, 10, 20, 25, 30, 40]
    assert len(ll) == 6


# Criterion 3
@pytest.mark.parametrize("bad", [-1, -5, 4, 100])
def test_out_of_range_raises_and_unchanged(bad):
    ll = build([1, 2, 3])
    with pytest.raises(IndexError):
        ll.insert(9, bad)
    assert ll.to_list() == [1, 2, 3]
    assert len(ll) == 3


def test_out_of_range_on_empty_list():
    ll = LinkedList()
    with pytest.raises(IndexError):
        ll.insert(1, 1)
    with pytest.raises(IndexError):
        ll.insert(1, -1)
    assert ll.to_list() == []
    assert len(ll) == 0


def test_list_still_usable_after_error():
    ll = build([1, 2])
    with pytest.raises(IndexError):
        ll.insert(9, 7)
    ll.insert(3)
    assert ll.to_list() == [1, 2, 3]
    assert len(ll) == 3


# Criterion 4
@pytest.mark.parametrize("bad", ["a", 1.5])
def test_non_int_index_raises_type_error(bad):
    ll = build([1, 2])
    with pytest.raises(TypeError):
        ll.insert(9, bad)
    assert ll.to_list() == [1, 2]
    assert len(ll) == 2


def test_non_int_index_other_types():
    ll = build([1])
    for bad in ([0], (0,), 1.0, "0"):
        with pytest.raises(TypeError):
            ll.insert(5, bad)
    assert ll.to_list() == [1]
    assert len(ll) == 1


def test_non_int_index_on_empty_list():
    ll = LinkedList()
    with pytest.raises(TypeError):
        ll.insert(1, "a")
    assert len(ll) == 0
    assert ll.to_list() == []


# Criterion 5
def test_insert_into_empty_at_zero():
    ll = LinkedList()
    ll.insert("x", 0)
    assert ll.to_list() == ["x"]
    assert len(ll) == 1


def test_insert_into_empty_then_more():
    ll = LinkedList()
    ll.insert(2, 0)
    ll.insert(1, 0)
    ll.insert(3, 2)
    assert ll.to_list() == [1, 2, 3]
    assert len(ll) == 3


# Criterion 6
def test_insert_falsy_values():
    ll = LinkedList()
    for v in (None, 0, "", False):
        ll.insert(v)
    assert ll.to_list() == [None, 0, "", False]
    assert len(ll) == 4


def test_many_appends_preserve_order():
    ll = LinkedList()
    for i in range(500):
        ll.insert(i)
    assert ll.to_list() == list(range(500))
    assert len(ll) == 500
