# Usage

All commands run from the repo root. With 1Password, prefix any command with `op run --env-file=.env.op --`. Otherwise, put your key in `.env` (copied from `.env.example`). The full settings table, the Python API and the cost-report schema are in the [README](../README.md#production-v4).

## CLI

```bash
uv run coder "Write fizzbuzz.py that prints FizzBuzz for 1..15, with pytest tests, and run them"
```

| Flag | Meaning |
|---|---|
| `task` | The coding task, in plain language. |
| `--model` | A provider-prefixed model: `anthropic:claude-haiku-5-5` (default since v4, was Sonnet), `anthropic:claude-sonnet-5-5`, `openai:gpt-4o`, `ollama:qwen2.5-coder`. Same as `CODER_MODEL`. |
| `--max-iterations N` | Model-call limit, from 1 to 10 (default 10). Values above 10 are rejected. Same as `CODER_MAX_ITERATIONS`. |
| `--timeout S` | `exec_python` timeout in seconds (default 10). Same as `EXEC_TIMEOUT_SECONDS`. |
| `--reset-memory` | Clears the sliding window, Chroma and the workspace. Can be used alone or before a task. |
| `--json` | Prints only the `CoderResult` JSON on stdout. Step events go to stderr. |
| `--demo` | Runs the two-task palindrome demo. The second task recalls the first from memory. |

Exit codes:
- **0**: every run succeeded.
- **1**: a run ended `failed`, `iteration_limit` or `error`.
- **2**: a usage or settings error, a busy workspace, a missing provider prerequisite, or Chroma unreachable (`CHROMA_HOST` set, server down).

## Demo

```bash
make demo        # = uv run coder --demo --reset-memory
```

- **Task 1:** `palindrome.py` plus pytest tests.
- **Task 2:** "Now make the palindrome checker also ignore punctuation…". The memory line before each call shows the first run retrieved from Chroma, and the agent reads and extends the existing files.

## Team (PM → Coder ↔ QA)

```bash
uv run team --demo api --reset-memory
uv run team --demo linked-list --reset-memory    # = make team-linked-list
uv run team --demo bst --reset-memory            # = make team-bst (the Docker default)
uv run team "Build a CLI that converts CSV to JSON, with tests"
```

| Flag | Meaning |
|---|---|
| `requirement` | The free-text requirement. Give this or `--demo` or `--reset-memory`, not a requirement together with `--demo`. |
| `--demo api` | The published grading task (GET /health, POST /echo with FastAPI). After the run it checks the generated API against the Postman fixture. |
| `--demo large` | An 11-feature to-do library requirement. The PM yields more than 8 tasks, so `consolidate_node` runs. |
| `--demo linked-list` | The Module 3 grading task (a linked list with insert, delete, search and reverse, all covered by tests). After the run it checks that every task was approved with tests run, that QA's canonical suites call all four methods, and that the project regression check passed. |
| `--demo bst` | The binary-search-tree grading task (`BinarySearchTree` with `insert`, `search` and `inorder`) plus an interface block titled "the supplied fixture's contract" (module `bst.py` at the workspace root; duplicate inserts ignored; values mutually comparable). After the run, the "BST fixture check" passes only if every task was approved with QA's tests run and the supplied fixture (`tests/fixtures/bst_fixture.py`, run with `--noconftest`) passes 5 of 5 tests. Coverage of `bst.py` under QA's tests is reported, not gated. |
| `--model` | The **Coder** model, provider-prefixed, as for `coder`. Same as `CODER_MODEL` (default `anthropic:claude-haiku-5-5`). |
| `--pm-model` | The PM model. Same as `PM_MODEL` (default `anthropic:claude-sonnet-5-5`). |
| `--qa-model` | The QA model. Same as `QA_MODEL` (default `anthropic:claude-haiku-5-5`). |
| `--max-iterations N` | Per-task Coder model-call limit, from 1 to 10 (default 10). |
| `--qa-max-iterations N` | QA test rounds per task, from 1 to 5 (default 5). Same as `QA_MAX_ITERATIONS`. A round is one QA test run, so 5 rounds means up to 4 Coder fix runs. A bad value (`0`, `6`, `abc`), from the flag or the variable, exits 2 with "invalid setting". |
| `--timeout S` | `exec_python` timeout in seconds (default 10). |
| `--reset-memory` | Clears both memory layers and the workspace first. Can be used alone. |
| `--json` | Prints only the final `ProjectState` JSON on stdout. Progress goes to stderr. |

Exit codes:
- **0**: the run `completed` (every task done, approved by QA and still passing the final regression check). With `--demo api`, `--demo linked-list` or `--demo bst`, that demo's check must also pass. `--reset-memory` alone also exits 0. The cost report's `outcome` is `success`.
- **1**: the run ended `partial` (cost-report `outcome: unresolved`; a task QA could not approve within the round limit, or one that regressed, with an `unresolved:` line per open issue) or `failed`, or a demo's check failed.
- **2**: a usage or settings error, a missing provider prerequisite, a busy workspace (another `coder` or `team` run holds the lock), or Chroma unreachable.
- **130**: interrupted with Ctrl-C (the lock is released; the cost report has `outcome: aborted`).

Each run writes its evidence to `.coder/runs/<run id>/`:

- `requirement.txt`, `state-after-pm.json`, `handoff-<task>.json`, `consolidation.json`
- `reflection-<task>.json`: the Coder's critique, before and after file hashes, changed files
- `qa-<task>-r<n>.json`: QA's report for each round (per-test results and issues); `qa-tests-<task>.py` is the suite
- `a2a-log.jsonl`: every Coder ↔ QA message sent, received or rejected
- `regression.json` (when at least one task is done): the end-of-run regression check
- `api-check.txt` or `demo-check.txt`, and `final-state.json`
- `events.jsonl`: the run's structured log, one JSON object per line (`ts`, `level`, `run_id`, `agent`, `event`, fields)
- `cost-report.json`: a copy of `docs/cost_reports/<run id>.json`; the terminal ends with `cost report: <path>`, or `cost report: unavailable (see events.jsonl)` if neither copy could be written

`state-after-<task>.json` and the `task_done` line in the terminal show a task's status when it closed. The regression check can later change a `done` task to `failed` (outcome `regressed`); the summary table, `regression.json` and `final-state.json` hold the final status.

`make team-demo`, `make team-large`, `make team-linked-list` and `make team-bst` run the four demos. They need a live model; run one with `op run --env-file=.env.op -- make team-bst`.

### Resilience, cache and logging settings

| Setting | Default | Meaning |
|---|---|---|
| `RETRY_MAX` | 3 | Retries after the original attempt, 0 to 3. Only transient errors (429, timeouts, 5xx and 529, connection errors) are retried, with full-jitter backoff. |
| `RETRY_BASE_DELAY_S` / `RETRY_MAX_DELAY_S` | 1.0 / 30.0 | Backoff: `uniform(0, min(max, base * 2**attempt))`. |
| `BREAKER_FAILURE_THRESHOLD` | 3 | Consecutive transient failures (after retries) that open the `pm`, `coder`, `qa`, `tools` or `chroma` breaker. |
| `BREAKER_COOLDOWN_S` | 30.0 | Wait before one half-open probe. |
| `CACHE_TTL_S` | 3600 | Response-cache TTL for structured PM and QA calls; `0` disables. The cache lives in `.coder/cache/` and survives `--reset-memory`. |
| `LOG_LEVEL` | INFO | Level of the JSON logs. |
| `CHROMA_HOST` / `CHROMA_PORT` | unset / 8000 | Use a Chroma server instead of the local `.chroma/` store. |
| `LANGSMITH_TRACING` + `LANGSMITH_API_KEY` | off | Both must be set for tracing. Each run is one `team-run:<run id>` trace; its URL is in the cost report's `trace_url` and the `trace` log event. |

To start a run without any cached PM or QA answers, delete `.coder/cache/` or set `CACHE_TTL_S=0`.

### QA settings

| Setting | Default | Meaning |
|---|---|---|
| `QA_MAX_ITERATIONS` / `--qa-max-iterations` | 5 | QA test rounds per task, 1 to 5. At the limit QA sends a `final_report`, and the task is `failed` with its unresolved issues; the run is `partial`. |
| `EXEC_TIMEOUT_SECONDS` / `--timeout` | 10 | Also bounds QA's pytest run. A timeout is a High issue, never an approval. |

To check an API you already generated in `workspace/`:

```bash
uv run python scripts/check_api.py                         # auto-detects `app = FastAPI(...)` in workspace/
uv run python scripts/check_api.py --app main:app --app-dir workspace --collection tests/fixtures/demo_api.postman_collection.json
```

It starts the app, runs each request in the Postman collection against it, prints one `PASS` / `FAIL` line per request, and exits 0 only if all pass.

## Web UI

```bash
uv run coder-web            # http://127.0.0.1:8000  (CODER_WEB_HOST / CODER_WEB_PORT)
```

- **Top:** a task box and Run, plus the `n/10` counter.
- **Left:** the workspace file tree. Files written in the current run are marked ●.
- **Centre:** a read-only code viewer that follows the most recently written file, with a terminal strip showing the last `exec_python` output.
- **Right:** the live agent timeline (memory, thought, plan, action, observation) and the final result card.

The UI is a viewer. It uses the same `CoderAgent` code path as the CLI. Requests whose `Host` header is not `127.0.0.1` or `localhost` get HTTP 400. This protects the code-running `/api/run` endpoint from DNS-rebinding pages. If you serve the UI under another name, set `CODER_WEB_ALLOWED_HOSTS` (comma-separated).

## Probes

**Timeout (C1.3).** The run is stopped by the timeout, and the observation says `timed_out=True (process tree killed)`.

```bash
uv run coder --timeout 2 "Write loop.py containing while True: pass and run it"
```

**Iteration limit (C1.4).** One model call is made. The tools it asks for are not executed, because the agent could never see their results, and the status is `iteration_limit`. Claude batches several actions per call, so a limit of 2–3 is often not reached on small tasks; `tests/test_graph.py` forces every limit with a scripted model, including the default 10.

```bash
uv run coder --max-iterations 1 "Write fizzbuzz.py that prints FizzBuzz for 1..15, with pytest tests, and run them"
uv run coder --max-iterations 11 "anything"     # rejected: must be between 1 and 10 (exit 2)
```

**Busy workspace.** Start a run, then start a second one in another terminal. It exits with code 2 and "workspace busy".

## Other providers

```bash
uv sync --extra openai   # or: pip install langchain-openai
OPENAI_API_KEY=... uv run coder --model openai:gpt-4o "task"

uv sync --extra ollama   # needs a running Ollama server with the model pulled
uv run coder --model ollama:qwen2.5-coder "task"
```

The live smoke tests use whatever `CODER_MODEL` selects (default Haiku 5.5). A skip message names the missing prerequisite.

## Tests

```bash
make test     # the mocked suite (addopts -m "not live"): no network, no paid calls
make verify   # REQUIRE_LIVE=1, live tests: real model calls cost money; a skipped live test is a failure
```
