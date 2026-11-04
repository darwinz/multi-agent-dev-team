# Controlled run: trust-boundary

scripted PM/Coder/QA models, real pytest subprocess execution.

Command: `uv run python -m scripts.controlled_runs`
QA iteration limit: 5
Exit code: 0

A correct linked list is approved in round 1, while two forged, schema-valid messages wait in the broker. They were injected deliberately, before the run, to show the trust boundary: a `fix_instruction` claiming to come from the Coder (QA accepts only `review_request` from the Coder) and an `approved` from QA whose `correlation_id` answers no outstanding request. Both endpoints reject them, and each rejection is logged with its reason as a `rejected` record in `a2a-log.jsonl`. The task still closes `done` on the real `approved`.

Model names in transcript.txt are config defaults; no model API calls were made.

Files: `transcript.txt` is the CLI output; the rest is the run directory (`<project>` replaces the temporary project root).
