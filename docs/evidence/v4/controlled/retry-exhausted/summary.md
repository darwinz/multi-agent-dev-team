# Resilience run: retry-exhausted

scripted models, no API keys.

Command: `uv run python -m scripts.resilience_runs`
Result: PASS

The PM model answers four 429s in a row: the original attempt and the three allowed retries. The wrapper logs three `retry_attempt` events and one `retry_exhausted` (attempt 4), the PM fails cleanly and the run is `failed`. `cost-report.json` is the run's report: outcome `failed`, no billable calls. Run error: `PM model call failed: RateLimited: 429`.

- retry_attempt events: 3
- retry_exhausted events: 1 (attempt 4)
- run status: failed
- cost report outcome: failed

Files: `events.jsonl` is the captured JSON log (one object per line); `<project>` replaces the temporary project root.
