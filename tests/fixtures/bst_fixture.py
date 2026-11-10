"""The supplied BST fixture (spec §5.8): the grading contract for the `bst` demo's generated module.

Run against a workspace with that workspace on PYTHONPATH; it is not collected by the default test
run (its name does not match test_*.py). Contract: `bst.py` at the workspace root defines
`BinarySearchTree` with insert(value) -> None, search(value) -> bool and inorder() -> list
(ascending); duplicate inserts are ignored.
"""

import random

from bst import BinarySearchTree


def make(values):
    tree = BinarySearchTree()
    for v in values:
        assert tree.insert(v) is None
    return tree


def test_insert_and_search_hit_and_miss():
    tree = make([50, 30, 70, 20, 40, 60, 80])
    for v in (50, 30, 70, 20, 40, 60, 80):
        assert tree.search(v) is True
    for v in (10, 35, 65, 90):
        assert tree.search(v) is False


def test_inorder_sorted_random():
    rng = random.Random(4)
    values = [rng.randint(-1000, 1000) for _ in range(200)]
    tree = make(values)
    assert tree.inorder() == sorted(set(values))
    assert all(tree.search(v) for v in values)


def test_inorder_sorted_degenerate():
    tree = make(range(300))  # sorted input: a fully unbalanced tree
    assert tree.inorder() == list(range(300))
    assert tree.search(299) is True and tree.search(300) is False


def test_duplicates_ignored():
    tree = make([5, 3, 5, 8, 3, 5])
    assert tree.inorder() == [3, 5, 8]
    assert tree.search(5) is True


def test_empty_tree():
    tree = BinarySearchTree()
    assert tree.search(1) is False
    assert tree.inorder() == []
