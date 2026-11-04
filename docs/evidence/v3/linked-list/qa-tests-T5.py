import ast
import pathlib

import pytest

from linked_list import LinkedList


def build(values):
    ll = LinkedList()
    for v in values:
        ll.insert(v)
    return ll


def check(ll, expected):
    assert ll.to_list() == list(expected)
    assert list(ll) == list(expected)
    assert len(ll) == len(expected)


# ---- criterion 1 ----

@pytest.mark.parametrize("values", [[], [1], [1, 2], [1, 2, 3], list(range(50)), list(range(1500))])
def test_reverse_lengths_and_returns_none(values):
    ll = build(values)
    result = ll.reverse()
    assert result is None
    check(ll, values[::-1])


def test_reverse_empty_stays_empty_and_usable():
    ll = LinkedList()
    assert ll.reverse() is None
    assert len(ll) == 0
    assert ll.to_list() == []
    assert ll.search(1) == -1
    ll.insert(1)
    check(ll, [1])


def test_reverse_single_element():
    ll = build(["a"])
    assert ll.reverse() is None
    check(ll, ["a"])
    assert ll.search("a") == 0


def test_reverse_two_elements():
    ll = build([1, 2])
    ll.reverse()
    check(ll, [2, 1])
    assert ll.search(2) == 0
    assert ll.search(1) == 1


def test_reverse_is_in_place_same_object():
    ll = build([1, 2, 3])
    ref = ll
    ll.reverse()
    assert ref is ll
    check(ref, [3, 2, 1])


def test_reverse_with_duplicates_and_falsy():
    vals = [None, 0, "", 0, None]
    ll = build(vals)
    ll.reverse()
    check(ll, vals[::-1])
    ll2 = build([0, None, "", False])
    ll2.reverse()
    assert ll2.to_list() == [False, "", None, 0]


# ---- criterion 2 ----

@pytest.mark.parametrize("values", [[], [1], [1, 2], [1, 2, 3, 4], list(range(100)), [None, 0, "", 0]])
def test_double_reverse_restores(values):
    ll = build(values)
    ll.reverse()
    ll.reverse()
    check(ll, values)


def test_double_reverse_then_append_goes_to_original_tail():
    ll = build([1, 2, 3])
    ll.reverse()
    ll.reverse()
    ll.insert(4)
    check(ll, [1, 2, 3, 4])


def test_triple_reverse_equals_single():
    ll = build([1, 2, 3, 4])
    for _ in range(3):
        ll.reverse()
    check(ll, [4, 3, 2, 1])


# ---- criterion 3 ----

def test_append_after_reverse_goes_to_new_tail():
    ll = build([1, 2, 3])
    ll.reverse()
    ll.insert(99)
    check(ll, [3, 2, 1, 99])


def test_append_after_reverse_not_at_old_tail_position():
    ll = build([1, 2, 3])
    ll.reverse()
    ll.insert(0)
    assert ll.to_list()[0] == 3
    assert ll.to_list()[-1] == 0


def test_insert_at_index_after_reverse():
    ll = build([1, 2, 3])
    ll.reverse()
    ll.insert("h", 0)
    ll.insert("m", 2)
    ll.insert("t", len(ll))
    check(ll, ["h", 3, "m", 2, 1, "t"])


def test_search_after_reverse():
    ll = build([10, 20, 30, 20])
    ll.reverse()  # 20,30,20,10
    assert ll.search(20) == 0
    assert ll.search(30) == 1
    assert ll.search(10) == 3
    assert ll.search(99) == -1


def test_delete_after_reverse_head_middle_tail():
    ll = build([1, 2, 3, 4, 5])
    ll.reverse()  # 5 4 3 2 1
    assert ll.delete(5) is True
    check(ll, [4, 3, 2, 1])
    assert ll.delete(2) is True
    check(ll, [4, 3, 1])
    assert ll.delete(1) is True
    check(ll, [4, 3])
    assert ll.delete(42) is False
    check(ll, [4, 3])


def test_delete_tail_after_reverse_then_append():
    ll = build([1, 2, 3])
    ll.reverse()  # 3 2 1
    ll.delete(1)
    ll.insert(7)
    check(ll, [3, 2, 7])


def test_delete_all_after_reverse_then_reverse_and_reuse():
    ll = build([1, 2, 3])
    ll.reverse()
    for v in (3, 2, 1):
        assert ll.delete(v) is True
    check(ll, [])
    assert ll.reverse() is None
    ll.insert(5)
    ll.insert(6)
    check(ll, [5, 6])


# ---- criterion 4 ----

def test_mixed_operations_against_list_model():
    ll = LinkedList()
    model = []

    def ins(v, idx=None):
        ll.insert(v, idx)
        model.insert(len(model) if idx is None else idx, v)

    def dele(v):
        expected = v in model
        assert ll.delete(v) is expected
        if expected:
            model.remove(v)

    def rev():
        assert ll.reverse() is None
        model.reverse()

    def verify():
        check(ll, model)
        for p in (0, 3, 7, 50, 99):
            assert ll.search(p) == (model.index(p) if p in model else -1)

    for i in range(10):
        ins(i)
        verify()
    rev(); verify()
    ins(50); verify()
    dele(3); verify()
    ins(99, 0); verify()
    rev(); verify()
    ins(7, 4); verify()
    dele(0); verify()
    dele(1000); verify()
    ins(100); verify()
    rev(); verify()
    ins(0, len(model)); verify()
    while model:
        dele(model[0]); verify()
    rev(); verify()
    ins(1); verify()


def test_len_after_reverse_and_mixed_ops():
    ll = build([1, 2, 3])
    ll.reverse()
    assert len(ll) == 3
    ll.insert(4)
    assert len(ll) == 4
    ll.delete(1)
    assert len(ll) == 3
    ll.delete(77)
    assert len(ll) == 3
    ll.reverse()
    assert len(ll) == 3
    assert sum(1 for _ in ll) == 3


def test_iteration_repeatable_after_reverse():
    ll = build([1, 2, 3])
    ll.reverse()
    assert list(ll) == list(ll) == [3, 2, 1]


def test_to_list_independent_copy_after_reverse():
    ll = build([1, 2])
    ll.reverse()
    a = ll.to_list()
    a.append(5)
    assert ll.to_list() == [2, 1]


def test_insert_errors_after_reverse_leave_list_unchanged():
    ll = build([1, 2, 3])
    ll.reverse()
    with pytest.raises(IndexError):
        ll.insert(9, 4)
    with pytest.raises(IndexError):
        ll.insert(9, -1)
    with pytest.raises(TypeError):
        ll.insert(9, "a")
    check(ll, [3, 2, 1])


# ---- criterion 5 ----

PUBLIC = {"insert", "search", "delete", "reverse", "to_list"}


def test_public_methods_are_the_expected_set():
    public = {n for n in dir(LinkedList)
              if not n.startswith("_") and callable(getattr(LinkedList, n))}
    assert PUBLIC <= public
    assert public == PUBLIC or public >= PUBLIC


def test_every_public_method_works_in_one_scenario():
    ll = LinkedList()
    assert ll.insert(1) is None
    ll.insert(2)
    ll.insert(0, 0)
    assert ll.search(2) == 2
    assert ll.delete(0) is True
    assert ll.reverse() is None
    assert ll.to_list() == [2, 1]
    assert len(ll) == 2
    assert list(iter(ll)) == [2, 1]
    assert "LinkedList" in repr(ll)


def test_delivered_test_suite_calls_every_public_method():
    path = pathlib.Path(__file__).with_name("test_linked_list.py")
    if not path.exists():
        pytest.skip("delivered test file not found")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    public = {n for n in dir(LinkedList)
              if not n.startswith("_") and callable(getattr(LinkedList, n))}
    assert public <= called
    test_names = [n.name for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    assert any("reverse" in n for n in test_names)
