# Resilience run: retry-then-success

scripted models, no API keys.

Command: `uv run python -m scripts.resilience_runs`
Result: PASS

The PM model answers one HTTP 429, then succeeds. The wrapper logs one `retry_attempt` (WARNING, with the jittered delay) and the `llm_call` records `attempts: 2`.

- retry_attempt events: 1
- llm_call attempts: 2

Files: `events.jsonl` is the captured JSON log (one object per line); `<project>` replaces the temporary project root.
