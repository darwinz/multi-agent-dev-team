# Controlled run: feedback

scripted PM/Coder/QA models, real pytest subprocess execution.

Command: `uv run python -m scripts.controlled_runs`
QA iteration limit: 5
Exit code: 0

Round 1 ships a linked list whose `reverse` is a no-op, so QA's `test_reverse` fails. QA's `fix_instruction` reaches the Coder, round 2 relinks the nodes, and the retest passes (`approved`).

Chain: `qa-T1-r1.json` (failing assertion) -> `a2a-log.jsonl` (`fix_instruction`) -> `a2a-log.jsonl` (`review_request` round 2, revised file hashes) -> `qa-T1-r2.json` (passing).

Model names in transcript.txt are config defaults; no model API calls were made.

Files: `transcript.txt` is the CLI output; the rest is the run directory (`<project>` replaces the temporary project root).
