# Resilience run: invalid-output

scripted models, no API keys.

Command: `uv run python -m scripts.resilience_runs`
Result: PASS

Two probes whose model returns invalid structured output (`scenario_step` lines are harness markers; each probe's section of `events.jsonl` follows its marker).

**pm-probe.** The PM returns two replies that are not a `Spec`. The run is `failed` (`PM produced an invalid spec twice: the model returned no structured spec`) and the section has zero `tool_call` events.

**qa-probe.** The Coder delivers normally (its own `write_file`/`exec_python` `tool_call` events come before QA's first model call and are legitimate). QA then returns two suites whose manifest names a test absent from the code. After QA's first `llm_call` there is no `tool_call` event: nothing was written or executed for QA. The task ends `qa_error`.

**What is logged for the rejection.** Production emits no dedicated WARNING event for an invalid PM spec or QA suite. The evidence of the rejection is the run outcome (`failed` for the PM, `qa_error` for the task) and `qa-report.json` (`qa_error`: `QA produced an invalid test suite twice: manifest test test_nope is not a top-level def test_* in the module; test test_push_then_pop is not in the manifest; map it to a criterion or remove it; test test_pop_returns_last is not in the manifest; map it to a criterion or remove it; criterion 2 ('delete(value) removes the first node holding value and returns True, or returns False if it is absent') has no test; criterion 3 ('search(value) returns True if value is in the list, otherwise False') has no test; criterion 4 ('reverse() reverses the list in place') has no test; If a criterion is about the tests themselves (for example "tests pass with pytest"), cover it with one short test that runs the Coder's delivered test file by its exact name (never your own test_qa_*.py or __file__) with subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", <file>], cwd=<the test's own directory>, timeout=5) and asserts the return code is 0`).

**Not claimed here.** The invalid *tool-call* probe (a `write_file` call with a missing or escaping argument rejected before any write) is covered by `tests/test_security.py::test_invalid_tool_call_rejected_before_write`, not by this run.

- pm-probe: run status failed, tool_call events: 0
- qa-probe: task outcome qa_error, tool_call events after QA's first llm_call: 0
- qa-probe: Coder tool_call events before QA (legitimate): 5
- WARNING/ERROR events in either run: 10

Files: `events.jsonl` is the captured JSON log (one object per line); `<project>` replaces the temporary project root.
