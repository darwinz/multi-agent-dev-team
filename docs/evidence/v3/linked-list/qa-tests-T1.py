import importlib
import os
import subprocess
import sys

import pytest

import linked_list
from linked_list import LinkedList, Node


# ---- Criterion 1: empty list behaviour ----

def test_empty_len_is_zero():
    assert len(LinkedList()) == 0


def test_empty_to_list_is_empty_list():
    result = LinkedList().to_list()
    assert result == []
    assert isinstance(result, list)


def test_empty_iter_yields_nothing():
    ll = LinkedList()
    assert list(iter(ll)) == []
    assert list(ll) == []
    with pytest.raises(StopIteration):
        next(iter(ll))


def test_empty_head_is_none():
    assert LinkedList().head is None


def test_separate_instances_independent():
    a = LinkedList()
    b = LinkedList()
    a.head = Node(1)
    a._size = 1
    assert len(b) == 0
    assert b.to_list() == []
    assert b.head is None


def test_to_list_returns_fresh_list():
    ll = LinkedList()
    a = ll.to_list()
    a.append(1)
    assert ll.to_list() == []


# ---- Node / non-empty structure (supports skeleton correctness) ----

def test_node_value_and_next_defaults():
    n = Node(5)
    assert n.value == 5
    assert n.next is None


def test_node_next_link():
    n = Node(5)
    m = Node(1, n)
    assert m.next is n
    assert m.value == 1


def test_nonempty_iteration_and_to_list_follow_head_chain():
    ll = LinkedList()
    ll.head = Node(1, Node(2, Node(3)))
    ll._size = 3
    assert len(ll) == 3
    assert ll.to_list() == [1, 2, 3]
    assert list(ll) == [1, 2, 3]
    # iterating twice gives same result
    assert list(ll) == [1, 2, 3]


def test_iter_yields_values_not_nodes():
    ll = LinkedList()
    ll.head = Node("a", Node("b"))
    ll._size = 2
    assert list(ll) == ["a", "b"]


def test_falsy_values_preserved():
    ll = LinkedList()
    ll.head = Node(0, Node(None, Node("", Node(False))))
    ll._size = 4
    assert ll.to_list() == [0, None, "", False]


# ---- Criterion 2: importability ----

def test_node_and_linkedlist_importable():
    mod = importlib.import_module("linked_list")
    assert isinstance(mod.Node, type)
    assert isinstance(mod.LinkedList, type)
    assert mod.Node is Node
    assert mod.LinkedList is LinkedList


def test_classes_instantiable():
    assert Node(1) is not None
    assert LinkedList() is not None


# ---- Criterion 3: empty-list tests exist and pass with pytest ----

def _delivered_test_path():
    return os.path.join(os.path.dirname(os.path.abspath(linked_list.__file__)),
                        "test_linked_list.py")


def test_delivered_test_file_exists_and_covers_empty_list():
    path = _delivered_test_path()
    assert os.path.isfile(path)
    with open(path, encoding="utf-8") as f:
        src = f.read()
    assert "def test_" in src
    assert "empty" in src.lower()
    assert "LinkedList" in src


def test_delivered_tests_pass_under_pytest(tmp_path):
    path = _delivered_test_path()
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.path.dirname(path) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", path, "-q", "-p", "no:cacheprovider",
         "--rootdir", str(tmp_path)],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
