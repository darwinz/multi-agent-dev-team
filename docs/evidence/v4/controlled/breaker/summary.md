# Resilience run: breaker

scripted models, no API keys.

Command: `uv run python -m scripts.resilience_runs`
Result: PASS

The `pm` breaker has threshold 3 and a real 0.2 s cool-down. Cycle 1: three failed calls open it, a fourth call is blocked (`call_blocked`, the model is not called), after the cool-down the probe succeeds and it closes. Cycle 2: three more failures open it again, and after the cool-down the probe fails and it reopens. Every transition is a WARNING `breaker_transition`.

- transitions: closed->open, open->half_open, half_open->closed, closed->open, open->half_open, half_open->open
- all transitions WARNING: yes

Files: `events.jsonl` is the captured JSON log (one object per line); `<project>` replaces the temporary project root.
