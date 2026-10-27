from agents.coder_agent import CoderAgent
from agents.result import CoderResult
from app.cli import main
from tests.fakes import FakeEmbedding, ScriptedChatModel, agent, ai, happy
from tools.workspace import RunLock


def test_json_output_is_valid_result(settings, capsys):
    code = main(["--json", "Write a palindrome checker"], agent=agent(settings, *happy()))
    out = capsys.readouterr().out
    assert code == 0 and CoderResult.model_validate_json(out).status == "success"


def test_max_iterations_above_cap_rejected(capsys):
    assert main(["--max-iterations", "11", "task"]) == 2
    assert "between 1 and 10" in capsys.readouterr().err


def test_limit_exits_1(settings):
    s = settings.model_copy(update={"max_iterations": 2})
    a = CoderAgent(s, model=ScriptedChatModel(responses=[ai("thinking") for _ in range(2)]),
                   embedding_function=FakeEmbedding())
    assert main(["task"], agent=a) == 1


def test_busy_exits_2(settings, capsys):
    with RunLock(settings.state_dir):
        assert main(["task"], agent=agent(settings)) == 2
    assert "workspace busy" in capsys.readouterr().err


def test_model_error_exits_1_without_traceback(settings, capsys):  # Review Focus 5
    assert main(["task"], agent=agent(settings, RuntimeError("401 invalid x-api-key"))) == 1
    err = capsys.readouterr()
    assert "Traceback" not in err.out + err.err and "401 invalid x-api-key" in err.out + err.err


def test_missing_key_exits_2(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CODER_PROJECT_DIR", str(tmp_path))
    assert main(["task"]) == 2 and "ANTHROPIC_API_KEY" in capsys.readouterr().err
