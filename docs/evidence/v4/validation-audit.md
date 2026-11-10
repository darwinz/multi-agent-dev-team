# Validation audit (spec §5.7)

Every place where LLM-produced text is acted on, and why none of it reaches a side-effecting tool
(`write_file`, `exec_python`, `run_action`) without validation. Line numbers refer to the committed code.

| # | Item | Where | Why it is safe | Proof |
|---|------|-------|----------------|-------|
| 1 | `extract_plan` regex over the Coder's free text | `orchestration/routing.py:15`, `:55`; used at `orchestration/graph.py:83` | The result only becomes the `plan` state field and a `plan` display event (`graph.py:85-86`, `finalize.py:50`). It is never passed to a tool or executed. | `tests/test_routing.py:39` `test_extract_plan`; `tests/test_security.py` `test_invalid_tool_call_rejected_before_write` (the Coder's free text has no path to `write_file`) |
| 2 | Free-text `fix` strings (QA fix plan, `Issue.fix`) interpolated into prompts | `orchestration/handoff.py:71` (`render_fix`), `agents/qa_agent.py:374` | The strings are Coder model input only. The Coder's resulting tool calls are validated like any other (row 6). `Issue.assertion` is length-capped (`orchestration/review.py:24`). | `tests/test_reflection.py:67` `test_render_fix_names_failure_fix_and_owned_file`; `tests/test_security.py` `test_invalid_tool_call_rejected_before_write` |
| 3 | `TestVerdict.corrected` (QA's replacement test code) | `agents/qa_agent.py:104` (`_replace_tests`), `:418`, `:422` | Spliced only if every replacement is exactly one function with the name it replaces and the module parses; the corrected suite must then pass `QATestSuite.check()` (`:422`). Otherwise the original suite is kept. The spliced file is written through `_checked_write_args` (`:236`, `:382`) and `guarded_tool`. | `tests/test_qa_agent.py:446` `test_invalid_correction_rejected` (renamed, extra test, unparsable); `tests/test_qa_agent.py:417` `test_wrong_test_corrected_in_round_one` |
| 4a | Silent fallback: PM consolidation | `agents/pm_agent.py:55-66` | The LLM merge plan is used only if `validate_merge` returns no problems (`:63`). Any exception falls back to `deterministic_merge`, and now logs WARNING `pm_consolidation_fallback` (fields `error_class`, `message`). | `tests/test_security.py` `test_consolidation_fallback_logs_warning`; `tests/test_consolidation.py:129` |
| 4b | Silent fallback: QA fix plan | `agents/qa_agent.py:366-377` | An exception keeps the code-generated fallback fixes and logs WARNING `qa_fix_plan_fallback`. | `tests/test_security.py` `test_fix_plan_and_audit_fallbacks_log_warning` |
| 4c | Silent fallback: QA test audit | `agents/qa_agent.py:388-413` | An exception keeps the suite unchanged and logs WARNING `qa_audit_fallback`. | `tests/test_security.py` `test_fix_plan_and_audit_fallbacks_log_warning` |
| 5 | Every `with_structured_output` call | PM `agents/pm_agent.py:32` (spec `:47-48`, merge `:62-63`); QA `agents/qa_agent.py:170` (suite `:202-206`, fix plan `:373`, audit `:410`); Coder critique `agents/coder_agent.py:92` (`:104-105`) | All use `include_raw=True`: the output is Pydantic-parsed; a `parsing_error` or `None` is treated as invalid (one repair retry for spec, suite, critique; fallback or `qa_error` otherwise). QA suites additionally pass `.check()` before they are written or run. | `tests/test_pm_agent.py:53` `test_two_invalid_raises`; `tests/test_qa_agent.py:169` `test_suite_failure_is_qa_error`; `tests/test_reflection.py:38`; `tests/test_security.py` `test_invalid_pm_spec_never_reaches_tools`, `test_invalid_qa_suite_never_executes` |
| 6 | Tool arguments | `tools/dispatch.py:37` (`ARG_MODELS[name].model_validate`) before `_dispatch`; QA's direct calls via `agents/qa_agent.py:236-239` (`_checked_write_args`, same `ARG_MODELS["write_file"]` plus the workspace check) and `guarded_tool` | Invalid or missing arguments raise `ToolError` and become an `error:` observation; the tool function is not called. A path outside the workspace is refused by `Workspace.resolve` inside `write_file`, before any write. | `tests/test_dispatch.py:42` `test_bad_calls_are_tool_errors`; `tests/test_qa_agent.py:465` `test_qa_direct_tool_args_validated`; `tests/test_security.py` `test_invalid_tool_call_rejected_before_write` |

## Invalid-output probe

`tests/test_security.py` spies on `tools.dispatch.write_file`/`exec_python` and `agents.qa_agent.write_file`/`exec_python`
(recorders that call through) and drives scripted runs:

- an invalid PM spec twice: run `failed`, "invalid spec twice", zero tool calls;
- two invalid QA suites: no QA `exec_python` call, the task closes with `qa_outcome == "qa_error"`;
- a Coder `write_file` with a path outside the workspace, then one without `content`: nothing written outside the
  workspace, both observations start with `error:`; the missing-content call never reaches `write_file`.
