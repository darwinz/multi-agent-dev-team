# Live run: linked-list grading demo

Real-model run (PM, Coder and QA on `anthropic:claude-sonnet-5-5`), real pytest subprocess execution.

Command: `op run --env-file=.env.op -- uv run team --demo linked-list --reset-memory`
QA iteration limit: 5
Run status: `completed`. Linked-list check: PASSED (`demo-check.txt`).

The PM split the requirement into five tasks (T1–T5). Every task was approved by QA's own tests; the final project regression check ran 109/109 QA tests over all five canonical suites (`regression.json`).

T1 shows the feedback loop with a live model: QA's round-1 tests failed, QA sent a `fix_instruction`, the Coder revised, and round 2 was approved. Chain: `qa-T1-r1.json` -> `a2a-log.jsonl` (`fix_instruction`, then `review_request` round 2 with revised hashes) -> `qa-T1-r2.json`. T2–T5 were approved in round 1.

Each task's `reflection-<id>.json` records the Coder's self-critique and the files its revision changed, before QA's first round.

Files: `transcript.txt` is the terminal output; the rest is the run directory. `<project>` replaces the local project path. The terminal's exit code was not captured (the `tee` target directory did not exist yet); the run status and check result above come from the transcript and `final-state.json`.
