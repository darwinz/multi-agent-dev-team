# Controlled run: nonconverging

scripted PM/Coder/QA models, real pytest subprocess execution.

Command: `uv run python -m scripts.controlled_runs`
QA iteration limit: 2
Exit code: 1

The Coder never fixes `reverse` and `QA_MAX_ITERATIONS=2`. At the limit QA sends a `final_report` listing the unresolved issue, the task is `failed` and the run is `partial`. No success is claimed.

Model names in transcript.txt are config defaults; no model API calls were made.

Files: `transcript.txt` is the CLI output; the rest is the run directory (`<project>` replaces the temporary project root).
