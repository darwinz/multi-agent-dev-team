import subprocess
import sys
from pathlib import Path

import pytest

from tests.fakes import agent, ai, happy, tool_call
from tools.workspace import RunLock, WorkspaceBusyError

REPO_ROOT = Path(__file__).resolve().parents[1]

RESET_SNIPPET = """
import sys
from agents.coder_agent import CoderAgent
from agents.config import Settings
from tests.fakes import FakeEmbedding
CoderAgent(Settings.from_env(project_dir=sys.argv[1]), model=object(), embedding_function=FakeEmbedding()).reset_memory()
"""


def test_run_returns_result_and_writes_memory(settings):
    a = agent(settings, *happy())
    r = a.run("Write a palindrome checker with tests")
    assert r.status == "success" and a.semantic.count() == 1 and len(a.window) == 1


def test_second_task_sees_first_in_memory(settings):
    # FakeEmbedding's distances aren't calibrated to the real model's 0.65 cutoff; this test checks plumbing
    settings = settings.model_copy(update={"memory_max_distance": 0.9})
    a = agent(settings, *happy(), ai("", tool_call("finish", explanation="e", plan=["p"])))
    a.run("Write a palindrome checker with tests")
    r = a.run("Now make the palindrome checker ignore punctuation")
    assert any("palindrome" in h.text for h in r.memories_used)
    assert "Write a palindrome checker" in a.model.calls[-1][0].content


def test_busy_workspace_raises_before_any_event(settings):
    with RunLock(settings.state_dir):
        # a fresh FileLock instance on the same path is a separate open file description, so it conflicts
        with pytest.raises(WorkspaceBusyError):
            next(agent(settings).stream("x"))


def test_reset_refused_while_run_holds_lock(settings):
    with RunLock(settings.state_dir):
        with pytest.raises(WorkspaceBusyError):
            agent(settings).reset_memory()


def test_model_error_becomes_error_result_and_releases_lock(settings):  # Review Focus 5
    a = agent(settings, RuntimeError("402 insufficient credits"))
    r = a.run("anything")
    assert r.status == "error" and "402 insufficient credits" in r.error and not r.verified
    with RunLock(settings.state_dir):
        pass


def test_reset_from_another_process_clears_window(settings):
    a = agent(settings, *happy(), ai("", tool_call("finish", explanation="e", plan=["p"])))
    a.run("Write a palindrome checker with tests")
    assert len(a.window) == 1
    subprocess.run([sys.executable, "-c", RESET_SNIPPET, str(settings.project_dir)], check=True, cwd=REPO_ROOT)
    a.run("next task")
    assert "Write a palindrome checker" not in a.model.calls[-1][0].content
    assert a.semantic.count() == 1  # only the run after the reset
