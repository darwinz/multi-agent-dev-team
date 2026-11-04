"""The PM → Coder handoff: build a TaskHandoff from project state and render it as Coder task text."""

from __future__ import annotations

from orchestration.a2a import FixInstruction
from orchestration.project_state import ProjectState, TaskHandoff
from orchestration.review import Critique

BUILD_ON = (
    "Build on the files from earlier tasks rather than replacing them, keep their tests passing, "
    "and run all tests before finishing: run each test file by its path with exec_python, one call "
    "per file. Tests run any other way (for example through subprocess) are not verified, so the "
    "task fails."
)


def qa_test_file(task_id: str) -> str:
    """The workspace-root file name of a task's QA suite, e.g. test_qa_t1.py."""
    return f"test_qa_{task_id.lower()}.py"


def make_handoff(state: ProjectState) -> TaskHandoff:
    """The handoff for tasks[current_task_index]; prior_files are the files earlier tasks produced."""
    tasks = state["tasks"]
    i = state["current_task_index"]
    task, spec = tasks[i], state["spec"]
    return TaskHandoff(
        task_id=task.task_id,
        title=task.title,
        description=task.description,
        acceptance_criteria=list(task.acceptance_criteria),
        priority=task.priority,
        position=i + 1,
        total=len(tasks),
        spec_name=spec.name,
        spec_description=spec.description,
        prior_files=sorted(state["code_output"]),
    )


def render_task(h: TaskHandoff) -> str:
    """The Coder's task text: built only from the handoff, never from the raw requirement."""
    files = "\n".join(f"- {p}" for p in h.prior_files) or "none yet"
    return "\n".join([
        f"Project: {h.spec_name}",
        h.spec_description,
        "",
        f"Task {h.task_id} ({h.position} of {h.total}, {h.priority} priority): {h.title}",
        h.description,
        "",
        "Acceptance criteria:",
        *(f"- [ ] {c}" for c in h.acceptance_criteria),
        "",
        "Files from earlier tasks:",
        files,
        "",
        BUILD_ON,
    ])


def render_revision(h: TaskHandoff, critique: Critique) -> str:
    """The task text plus the Coder's own critique, to be addressed before QA sees the code."""
    items = [f"- [{i.kind}] {i.location}: {i.concern} → {i.change}" for i in critique.items]
    return "\n".join([
        render_task(h), "", "Self-review:", *items, "",
        "Address every item with a code change or a new test, keep all tests passing, "
        "and run each test file with exec_python before finishing.",
    ])


def render_fix(h: TaskHandoff, fix: FixInstruction) -> str:
    """The task text plus QA's failures; the QA suite file belongs to QA."""
    qa_file = qa_test_file(h.task_id)
    lines = [f"{i.test} ({i.criterion}): {i.assertion} → fix: {i.fix}" for i in fix.issues]
    return "\n".join([
        render_task(h), "", f"QA review round {fix.round} failed:", *lines, "",
        f"QA owns {qa_file}; do not modify it. Fix the code, then run {qa_file} and your own tests with exec_python.",
    ])
