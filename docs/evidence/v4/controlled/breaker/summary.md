# Resilience run: breaker

scripted models, no API keys.

Command: `uv run python -m scripts.resilience_runs`
Result: PASS

The `pm` breaker with threshold 3 and a real 0.2 s cool-down, in two legs (`scenario_step` and `call_blocked` lines are harness annotations, not production events).

**Leg 1, retry composed with the breaker (default `retry_max=3`).** Three calls each make 4 attempts (12 model calls in all, 3 `retry_attempt` then `retry_exhausted` at attempt 4 per call). Each exhausted call is one breaker failure (1, 2, 3), so the third opens the breaker (closed->open). A fourth call is blocked without reaching the model: the model call count stays at 12.

**Leg 2, half-open cycles (`retry_max=0`, only to keep the run short: one attempt per call).** Cycle 1: three failures open it, a call while open is blocked, after the cool-down the probe succeeds and it closes. Cycle 2: three more failures open it, and after the cool-down the probe fails and it reopens. Every `breaker_transition` is a WARNING.

- leg 1 retry_attempt events: 9, retry_exhausted events: 3 (attempt 4 each)
- leg 1 model calls: 12 before the breaker opened, 12 after the blocked call
- leg 1 breaker failures after each call: 1, 2, 3 -> closed->open
- leg 2 transitions: closed->open, open->half_open, half_open->closed, closed->open, open->half_open, half_open->open
- all transitions WARNING: yes

Files: `events.jsonl` is the captured JSON log (one object per line); `<project>` replaces the temporary project root.
