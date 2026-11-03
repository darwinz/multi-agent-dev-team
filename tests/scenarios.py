"""Controlled scenarios (spec §12): scripted PM, Coder and QA models; pytest runs for real.

feedback: round 1 ships a linked list whose `reverse` is a no-op, QA's test fails, the fix reaches
the Coder, round 2 fixes it and QA approves. nonconverging: the Coder never fixes `reverse`.
"""

from __future__ import annotations

from typing import Any

from agents.config import Settings
from agents.pm_agent import PMAgent
from orchestration.project_state import AcceptanceCriterion, Spec, Task
from orchestration.team_runner import TeamRunner
from tests.fakes import (
    ScriptedChatModel,
    agent,
    ai,
    critique_response,
    fix_plan_response,
    qa_agent,
    qa_suite_response,
    spec_response,
    tool_call,
)

LL_CRITERIA = [
    "insert(value) appends value to the end of the list",
    "delete(value) removes the first node holding value and returns True, or returns False if it is absent",
    "search(value) returns True if value is in the list, otherwise False",
    "reverse() reverses the list in place",
]

_LL_BODY = '''\
class Node:
    def __init__(self, value, next=None):
        self.value = value
        self.next = next


class LinkedList:
    def __init__(self):
        self.head = None

    def insert(self, value):
        node = Node(value)
        if self.head is None:
            self.head = node
            return
        cur = self.head
        while cur.next is not None:
            cur = cur.next
        cur.next = node

    def delete(self, value):
        prev, cur = None, self.head
        while cur is not None:
            if cur.value == value:
                if prev is None:
                    self.head = cur.next
                else:
                    prev.next = cur.next
                return True
            prev, cur = cur, cur.next
        return False

    def search(self, value):
        cur = self.head
        while cur is not None:
            if cur.value == value:
                return True
            cur = cur.next
        return False

    def to_list(self):
        out, cur = [], self.head
        while cur is not None:
            out.append(cur.value)
            cur = cur.next
        return out
'''

BUGGY_LL = _LL_BODY + '''
    def reverse(self):
        pass  # bug: a no-op
'''

FIXED_LL = _LL_BODY + '''
    def reverse(self):
        prev, cur = None, self.head
        while cur is not None:
            cur.next, prev, cur = prev, cur, cur.next
        self.head = prev
'''

# The reflection revision: documented, but `reverse` is still a no-op
BUGGY_LL_DOCUMENTED = '"""A singly linked list."""\n\n' + BUGGY_LL

CODER_LL_TESTS = '''\
from linked_list import LinkedList


def test_insert_and_search():
    ll = LinkedList()
    ll.insert(1)
    ll.insert(2)
    assert ll.search(2)
    assert not ll.search(3)
'''

QA_LL_TESTS = '''\
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
'''

QA_LL_MANIFEST = (("test_insert", 1), ("test_delete", 2), ("test_search", 3), ("test_reverse", 4))


def linked_list_spec() -> Spec:
    return Spec(
        name="Linked list",
        description="A singly linked list with insert, delete, search and reverse.",
        acceptance_criteria=[AcceptanceCriterion(id="AC1", text="The linked list supports insert, delete, "
                                                                "search and reverse")],
        tasks=[Task(task_id="T1", title="Linked list", description="Implement LinkedList in linked_list.py.",
                    acceptance_criteria=list(LL_CRITERIA), priority="high", covers=["AC1"])],
    )


def coder_run(files: dict[str, str], test: str, explanation: str) -> list[Any]:
    """One scripted Coder run: write the files, run `test`, finish."""
    writes = [tool_call("write_file", path=p, content=c) for p, c in files.items()]
    return [
        ai("write and run", *writes, tool_call("exec_python", path=test)),
        ai("done", tool_call("finish", explanation=explanation, plan=["write", "run"], entry_point=test)),
    ]


def _deliver_and_reflect() -> list[Any]:
    return [
        *coder_run({"linked_list.py": BUGGY_LL, "test_linked_list.py": CODER_LL_TESTS}, "test_linked_list.py",
                   "Linked list with insert, delete, search and reverse"),
        critique_response(("edge_case", "linked_list.py:LinkedList", "the class has no documentation",
                           "add a module docstring")),
        *coder_run({"linked_list.py": BUGGY_LL_DOCUMENTED}, "test_linked_list.py", "Documented the module"),
    ]


def _reverse_fix() -> Any:
    return fix_plan_response(("test_reverse", "reverse() is a no-op",
                              "relink each node to its predecessor and point head at the old tail"))


def feedback_scripts() -> tuple[list, list, list]:
    """(pm, coder, qa): round 1 fails on test_reverse, round 2 is fixed and approved."""
    coder = [*_deliver_and_reflect(),
             *coder_run({"linked_list.py": FIXED_LL}, "test_linked_list.py", "Fixed reverse")]
    qa = [qa_suite_response(QA_LL_TESTS, QA_LL_MANIFEST), _reverse_fix()]
    return [spec_response(linked_list_spec())], coder, qa


def nonconverging_scripts(rounds: int) -> tuple[list, list, list]:
    """(pm, coder, qa): `reverse` is never fixed, so QA ends at `rounds` with final_report."""
    coder = [*_deliver_and_reflect()]
    for n in range(1, rounds):
        coder += coder_run({"linked_list.py": BUGGY_LL_DOCUMENTED + f"\n# attempt {n}\n"},
                           "test_linked_list.py", f"Attempt {n} at reverse")
    qa = [qa_suite_response(QA_LL_TESTS, QA_LL_MANIFEST), *(_reverse_fix() for _ in range(rounds))]
    return [spec_response(linked_list_spec())], coder, qa


def scripted_runner(settings: Settings, scripts: tuple[list, list, list]) -> TeamRunner:
    """A TeamRunner whose PM, Coder (FakeEmbedding memory) and QA answer from (pm, coder, qa) scripts."""
    pm, coder, qa = scripts
    return TeamRunner(settings, pm=PMAgent(settings, model=ScriptedChatModel(responses=list(pm))),
                      coder=agent(settings, *coder), qa=qa_agent(settings, *qa))
