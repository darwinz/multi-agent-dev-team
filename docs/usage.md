# Usage

All commands run from the repo root. With 1Password, prefix any command with `op run --env-file=.env.op --`. Otherwise, put your key in `.env` (copied from `.env.example`).

## CLI

```bash
uv run coder "Write fizzbuzz.py that prints FizzBuzz for 1..15, with pytest tests, and run them"
```

| Flag | Meaning |
|---|---|
| `task` | The coding task, in plain language. |
| `--model` | A provider-prefixed model: `anthropic:claude-sonnet-5-5` (default), `openai:gpt-4o`, `ollama:qwen2.5-coder`. Same as `CODER_MODEL`. |
| `--max-iterations N` | Model-call limit, from 1 to 10 (default 10). Values above 10 are rejected. Same as `CODER_MAX_ITERATIONS`. |
| `--timeout S` | `exec_python` timeout in seconds (default 10). Same as `EXEC_TIMEOUT_SECONDS`. |
| `--reset-memory` | Clears the sliding window, Chroma and the workspace. Can be used alone or before a task. |
| `--json` | Prints only the `CoderResult` JSON on stdout. Step events go to stderr. |
| `--demo` | Runs the two-task palindrome demo. The second task recalls the first from memory. |

Exit codes:
- **0**: every run succeeded.
- **1**: a run ended `failed`, `iteration_limit` or `error`.
- **2**: a usage or settings error, a busy workspace, or a missing provider prerequisite.

## Demo

```bash
make demo        # = uv run coder --demo --reset-memory
```

- **Task 1:** `palindrome.py` plus pytest tests.
- **Task 2:** "Now make the palindrome checker also ignore punctuation…". The memory line before each call shows the first run retrieved from Chroma, and the agent reads and extends the existing files.

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

The live smoke tests use whatever `CODER_MODEL` selects. A skip message names the missing prerequisite.

## Tests

```bash
make test     # unit tests; live tests skip without a provider
make verify   # REQUIRE_LIVE=1: a skipped live test is a failure
```
