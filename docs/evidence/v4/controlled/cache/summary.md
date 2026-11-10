# Resilience run: cache

scripted models, no API keys.

Command: `uv run python -m scripts.resilience_runs`
Result: PASS

A PM spec request with a 2 s response-cache TTL. First call: `cache_miss`, the model answers. Same request again: `cache_hit` (zero-cost `llm_call`, `attempts: 0`, the model is not called). After sleeping 2.1 s: `cache_expired`, the model is called again.

- cache events: cache_miss, cache_hit, cache_expired
- model calls: 2 for 3 requests

Files: `events.jsonl` is the captured JSON log (one object per line); `<project>` replaces the temporary project root.
