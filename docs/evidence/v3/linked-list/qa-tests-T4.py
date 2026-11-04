import inspect

import pytest

from linked_list import LinkedList, Node


def build(values):
    ll = LinkedList()
    head = None
    for v in reversed(values):
        head = Node(v, head)
    ll.head = head
    ll._size = len(values)
    return ll


def chain_count(ll):
    n = 0
    node = ll.head
    while node is not None:
        n += 1
        node = node.next
    return n


def check(ll, expected):
    assert ll.to_list() == expected
    assert list(ll) == expected
    assert len(ll) == len(expected)
    assert chain_count(ll) == len(expected)


# ---- criterion 1 ----

def test_delete_head_returns_true():
    ll = build([1, 2, 3])
    assert ll.delete(1) is True
    check(ll, [2, 3])


def test_delete_middle_returns_true():
    ll = build([1, 2, 3, 4])
    assert ll.delete(3) is True
    check(ll, [1, 2, 4])


def test_delete_tail_returns_true_and_append_works():
    ll = build([1, 2, 3])
    assert ll.delete(3) is True
    check(ll, [1, 2])
    ll.insert(9)
    check(ll, [1, 2, 9])


def test_delete_only_element_returns_true():
    ll = build(["a"])
    assert ll.delete("a") is True
    assert ll.head is None
    check(ll, [])
    assert not ll


def test_delete_using_insert_built_list():
    ll = LinkedList()
    for v in [10, 20, 30]:
        ll.insert(v)
    assert ll.delete(20) is True
    check(ll, [10, 30])
    assert ll.delete(10) is True
    assert ll.delete(30) is True
    check(ll, [])


def test_delete_falsy_values():
    ll = build([None, 0, "", None])
    assert ll.delete(None) is True
    check(ll, [0, "", None])
    assert ll.delete("") is True
    check(ll, [0, None])
    assert ll.delete(0) is True
    check(ll, [None])
    assert ll.delete(None) is True
    check(ll, [])


# ---- criterion 2 ----

def test_len_decreases_by_one_each_delete():
    ll = build([1, 2, 3, 4])
    expected = 4
    for v in (2, 4, 1, 3):
        assert ll.delete(v) is True
        expected -= 1
        assert len(ll) == expected
        assert chain_count(ll) == expected


def test_len_after_single_delete_from_large_list():
    ll = build(list(range(200)))
    before = len(ll)
    assert ll.delete(150) is True
    assert len(ll) == before - 1
    assert 150 not in ll.to_list()


def test_len_zero_after_deleting_only_element_then_reuse():
    ll = build([5])
    ll.delete(5)
    assert len(ll) == 0
    ll.insert(6)
    check(ll, [6])


# ---- criterion 3 ----

def test_only_first_duplicate_removed():
    ll = build([2, 1, 2, 2])
    assert ll.delete(2) is True
    check(ll, [1, 2, 2])
    assert ll.delete(2) is True
    check(ll, [1, 2])


def test_duplicate_in_middle_first_occurrence_removed():
    ll = build([1, 5, 2, 5, 3])
    assert ll.delete(5) is True
    check(ll, [1, 2, 5, 3])


def test_all_duplicates_removed_one_at_a_time():
    ll = build([7, 7, 7])
    for remaining in (2, 1, 0):
        assert ll.delete(7) is True
        assert len(ll) == remaining
    assert ll.delete(7) is False


def test_first_equal_node_object_removed():
    a = tuple([1, 2])
    b = tuple([1, 2])
    ll = build([a, 0, b])
    first = ll.head
    assert ll.delete(tuple([1, 2])) is True
    assert ll.head is not first
    assert ll.head.value == 0
    assert ll.head.next.value is b
    assert len(ll) == 2


# ---- criterion 4 ----

def test_missing_value_returns_false_unchanged():
    ll = build([1, 2, 3])
    nodes = [ll.head, ll.head.next, ll.head.next.next]
    assert ll.delete(99) is False
    check(ll, [1, 2, 3])
    assert ll.head is nodes[0]
    assert nodes[0].next is nodes[1]
    assert nodes[1].next is nodes[2]
    assert nodes[2].next is None


def test_empty_list_returns_false():
    ll = LinkedList()
    assert ll.delete(1) is False
    assert ll.delete(None) is False
    assert ll.head is None
    check(ll, [])


def test_delete_twice_second_false():
    ll = build([1, 2])
    assert ll.delete(1) is True
    assert ll.delete(1) is False
    check(ll, [2])


def test_missing_does_not_raise_for_odd_types():
    ll = build([1, 2, 3])
    for probe in (None, "1", [1], 4.5, object()):
        assert ll.delete(probe) is False
    check(ll, [1, 2, 3])


def test_missing_single_element_list():
    ll = build([1])
    assert ll.delete(2) is False
    check(ll, [1])


def test_delete_returns_bool_type():
    ll = build([1])
    assert type(ll.delete(2)) is bool
    assert type(ll.delete(1)) is bool


# ---- criterion 5 ----

def test_delete_exists_and_is_callable():
    assert callable(getattr(LinkedList, "delete", None))
    params = list(inspect.signature(LinkedList.delete).parameters)
    assert params[:2] == ["self", "value"]


def test_delivered_delete_tests_pass():
    import test_linked_list as delivered
    names = [n for n in dir(delivered) if n.startswith("test_delete")]
    assert names, "no delete tests delivered"
    for n in names:
        fn = getattr(delivered, n)
        if inspect.signature(fn).parameters:
            continue
        fn()
