from linked_list import LinkedList


def make(*values):
    ll = LinkedList()
    for v in values:
        ll.insert(v)
    return ll


def test_insert():
    assert make(1, 2, 3).to_list() == [1, 2, 3]


def test_delete():
    ll = make(1, 2, 3)
    assert ll.delete(2) is True
    assert ll.to_list() == [1, 3]
    assert ll.delete(9) is False


def test_search():
    ll = make(1, 2)
    assert ll.search(2) is True
    assert ll.search(5) is False


def test_reverse():
    ll = make(1, 2, 3)
    ll.reverse()
    assert ll.to_list() == [3, 2, 1]
