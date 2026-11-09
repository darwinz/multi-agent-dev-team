# Resilience run: invalid-output

scripted models, no API keys.

Command: `uv run python -m scripts.resilience_runs`
Result: PASS

The Task 13 probes. PM probe: the PM model returns two replies that are not a `Spec`; the run fails (`PM produced an invalid spec twice: the model returned no structured spec`) with no `tool_call` event at all. QA probe: the Coder delivers normally, then QA returns two suites whose manifest names a test absent from the code; the task ends `qa_error` and QA never executes anything (no `tool_call` event from the `qa` agent). `events.jsonl` holds the PM run's log followed by the QA run's.

- PM probe tool_call events: 0
- QA probe tool_call events from qa agent: 0
- QA probe tool_call events from the Coder (its normal delivery): 5
- QA probe task outcome: qa_error

Files: `events.jsonl` is the captured JSON log (one object per line); `<project>` replaces the temporary project root.
