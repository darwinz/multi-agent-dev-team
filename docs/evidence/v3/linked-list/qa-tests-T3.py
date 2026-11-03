import os
import subprocess
import sys
from pathlib import Path

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


# ---- Criterion 1: head, middle, tail ----

def test_search_head():
    assert build([10, 20, 30, 40]).search(10) == 0


def test_search_middle():
    ll = build([10, 20, 30, 40])
    assert ll.search(20) == 1
    assert ll.search(30) == 2


def test_search_tail():
    assert build([10, 20, 30, 40]).search(40) == 3


def test_search_single_element_list():
    ll = build(["a"])
    assert ll.search("a") == 0
    assert ll.search("b") == -1


def test_search_all_positions_in_longer_list():
    values = list(range(100, 150))
    ll = build(values)
    for i, v in enumerate(values):
        assert ll.search(v) == i


def test_search_works_on_list_built_with_insert():
    ll = LinkedList()
    for v in ["x", "y", "z"]:
        ll.insert(v)
    assert ll.search("x") == 0
    assert ll.search("y") == 1
    assert ll.search("z") == 2


def test_search_non_numeric_values_and_none():
    ll = build([None, "s", (1, 2), 3.5])
    assert ll.search(None) == 0
    assert ll.search("s") == 1
    assert ll.search((1, 2)) == 2
    assert ll.search(3.5) == 3


def test_search_does_not_modify_list():
    ll = build([1, 2, 3])
    head = ll.head
    ll.search(3)
    ll.search(42)
    assert ll.head is head
    assert ll.to_list() == [1, 2, 3]
    assert len(ll) == 3


# ---- Criterion 2: duplicates ----

def test_search_returns_first_of_duplicates():
    ll = build([1, 2, 2, 3, 2])
    assert ll.search(2) == 1


def test_search_duplicates_at_head_and_all_equal():
    assert build([7, 7]).search(7) == 0
    assert build([5, 5, 5, 5]).search(5) == 0


def test_search_duplicate_at_head_and_tail():
    assert build([9, 1, 2, 9]).search(9) == 0


def test_search_duplicates_only_tail_two():
    assert build([1, 2, 3, 3]).search(3) == 2


# ---- Criterion 3: missing / empty ----

def test_search_missing_returns_minus_one():
    ll = build([1, 2, 3])
    assert ll.search(4) == -1
    assert ll.search(0) == -1
    assert ll.search("1") == -1


def test_search_empty_list_returns_minus_one():
    assert LinkedList().search(1) == -1
    assert LinkedList().search(None) == -1


def test_search_none_missing_in_nonempty():
    assert build([1, 2]).search(None) == -1


def test_search_missing_after_removal_by_hand():
    ll = build([1, 2, 3])
    ll.head.next.next = None
    ll._size = 2
    assert ll.search(3) == -1
    assert ll.search(2) == 1


def test_search_returns_int():
    r = build([1]).search(1)
    assert isinstance(r, int) and not isinstance(r, bool)


# ---- Criterion 4: tests pass ----

def test_delivered_test_suite_passes(tmp_path):
    here = Path(__file__).resolve().parent
    target = here / "test_linked_list.py"
    if not target.exists():
        pytest.skip("delivered test file not found")
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(here) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         str(target)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_delivered_tests_include_search_tests():
    here = Path(__file__).resolve().parent
    target = here / "test_linked_list.py"
    if not target.exists():
        pytest.skip("delivered test file not found")
    text = target.read_text()
    assert "search(" in text
    assert "def test_" in text
