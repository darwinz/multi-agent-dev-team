# Coder Agent v1.0 — Design

**Course:** Saras AI Institute, *Build Autonomous Multi-Agent Systems* (Aug 2026), Module 1 course project
**Author of requirements:** Brandon Johnson (student). Claude writes the code; Brandon reviews.
**Date:** 2026-10-25

> **Sequencing (added 2026-10-25):** the LMS assignment page says four lab repos
> (`SarasAI-Institute/Multi-agents-lab-1-1-memory`, `-1-2-tools`, `-1-3-react-coder`,
> `-1-4-langgraph-autogen-crewai`) "collectively form the Week 1 deliverable" and asks for the
> forked repo URL. The labs are completed first, in their forks, exactly to their own contracts
> (course `llm_client.py`, `gpt-4.1-mini` via Helicone, their tests). This design is the second
> step: the combined portfolio repo, built on what the labs produce. Open question for course
> staff: is the graded submission the lab forks or this combined repo?

## 1. Intent

Module 1 of a five-module capstone. The end state (Module 5) is an autonomous software team:
Product Manager agent → Coder agent → QA & Debugger agent. Module 1 is the Coder agent alone,
and every later module builds on it, so it has to be solid.

Two goals, equal weight:

1. **Pass all six Module 1 rubric criteria cleanly** (C1.1–C1.6, below). The assessor clones the
   repo, runs setup, runs the agent, and *probes* it (forces an exec timeout, forces the
   iteration limit). Everything has to be real, not simulated.
2. **Portfolio-worthy.** A public repo that a recruiter or engineering manager can clone, run
   with one command, and watch an agent build software — in the terminal and in a web UI.

## 2. Source requirements (verbatim intent from the course)

### Project guide, Module 1 instructions
1. Initialise the GitHub repo with the required folder structure (§5). Commit an initial README.
2. Memory with two layers: a sliding-window conversation buffer (last N turns) and a
   Chroma-backed semantic memory store. The agent must embed and retrieve relevant past context
   **before every LLM call**.
3. Two custom tools used throughout the course: `read_file` / `write_file` for file I/O, and
   `exec_python` for sandboxed code execution using `subprocess` with a configurable timeout.
4. A ReAct loop (Thought → Action → Observation → repeat) that uses those tools to decompose and
   solve a multi-step coding task, with a loop-detection counter capped at **10 iterations**.
5. Structured output containing: generated code, natural-language explanation, the execution plan
   used, and the result of running the code.
6. At minimum a basic smoke test that invokes the agent with a short coding task and asserts the
   structured output has the correct fields.
7. README with setup instructions and a brief description of implementation decisions.

Technical requirements: framework LangGraph / AutoGen / CrewAI (→ **LangGraph**); LLM GPT-4o,
Claude Sonnet, or Ollama (→ **Claude Sonnet**); memory backend Chroma (required); code execution
via subprocess with a 10-second default timeout, **no direct `eval()`**; structured output as a
Pydantic model with at least `code`, `explanation`, `plan`, `result`.

Submission: push to GitHub; submit repo link **and commit SHA** in the LMS. 2 attempts.

### Rubric (1 point each, Met / Not Yet)
| ID | Met when | Not Yet when |
|---|---|---|
| C1.1 Reproducible project | Public repo has `agents/ tools/ orchestration/ tests/ docs/`, `.env.example`, `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `README.md` (future parts may be placeholders). README explains setup, chosen framework, implementation decisions; setup reaches a runnable Coder. | Scaffolding missing or setup can't reach the agent. |
| C1.2 Memory | Sliding window keeps last N turns. Chroma store embeds task context and retrieves before **every** LLM call; retrieved context is supplied to the agent. A related follow-up demonstrates retrieval. Empty first-run retrieval is fine. | Memory only declared, one layer only, or retrieval never reaches the LLM. |
| C1.3 Tools | Agent *actually* invokes `read_file`, `write_file`, uses `exec_python` in a subprocess. Configurable timeout, 10 s default, no `eval()`. A non-terminating execution is stopped by the timeout (assessor probes this). | Manual file changes, simulated execution, `eval()`, or timeout fails. |
| C1.4 Bounded ReAct | Accepts a natural-language task; ReAct action/observation loop decomposes and solves it with tools; observed results drive next actions; counter stops after **at most 10** iterations (assessor forces this). | Single unconnected LLM call, tools don't affect execution, or loop can exceed 10. |
| C1.5 Validated output | Pydantic-validated output with `code`, `explanation`, `plan`, `result`. **Code matches the artifact written by the tools; result reflects actual execution**; explanation/plan describe the task. Assessor inspects the validation path. | Missing field, validation bypassed, success claimed without execution. |
| C1.6 Verified | Automated smoke test invokes the Coder with a short task and asserts the structured fields; it passes; an observed run produces working code with no human repair. | Test checks a fixed object, fields not asserted, or generated code fails. |

## 3. Goals, non-goals, anti-goals

**Goals**
- All six rubric criteria demonstrably Met, with evidence easy for an assessor to find
  (README section mapping each criterion to code + how to observe it).
- One-command first run that shows the agent working, in the terminal.
- A local web UI ("mini-IDE") showing the same run live.
- Clean, modular code that Modules 2–5 can extend (PM handoff, QA loop, MCP/A2A, Docker,
  observability) without rewrites.

**Non-goals (Module 1)**
- PM and QA agents (placeholders only).
- MCP / A2A integration, cost optimisation, production observability (later modules).
- A true security sandbox (container/VM isolation). Module 1 is subprocess + timeout + workspace
  confinement; stronger isolation arrives with Docker in Module 4. README states this honestly.
- Editing code in the web UI; multi-user or hosted deployment of the UI.

**Anti-goals (failure even if it "works")**
- Any faked evidence: code in the output that differs from what was written to disk, a `result`
  not produced by actually running code, memory that is stored but never reaches the prompt.
- `eval()` / `exec()` of generated code in-process.
- A loop path that can exceed 10 iterations.
- Secrets committed to the public repo.
- The rubric depending on the web UI (UI is a viewer; CLI + tests carry the grade).

## 4. Decisions made by Brandon
- Public GitHub repo (the course's private SarasAI starter repo is not used for this project).
- LangGraph.
- Claude Sonnet via Brandon's own Anthropic key.
- Both a terminal walkthrough and a web UI.
- Web UI layout **B: mini-IDE** — task input + Run across the top; left: workspace file tree
  (files appear as the agent writes them); centre: read-only code viewer with a terminal strip
  for run/test output; right: live agent panel (Thought / Action / Observation / Memory steps,
  iteration counter "n/10"); final structured result shown on completion.
- Terminal first-run experience: live step-by-step output, then a result panel; followed by a
  related follow-up task that visibly recalls the first from memory.
- Claude writes the code; Brandon reviews.

## 5. Constraints

### Required repository structure (from the project guide, §1.5)
```
multi-agent-dev-team/
├── agents/            # One module per agent
│   ├── pm_agent.py        (placeholder in M1)
│   ├── coder_agent.py
│   └── qa_agent.py        (placeholder in M1)
├── tools/             # Shared tool definitions
├── orchestration/     # LangGraph graph
├── tests/             # Unit, integration, E2E
├── docs/              # Architecture diagrams, usage guides
├── .env.example       # Env var template, no secrets
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md          # Setup, usage, architecture overview
```
Additional top-level items are allowed (e.g. `pyproject.toml`, `uv.lock`, `web/`, `Makefile`,
`.env.op`). These top-level dirs are importable packages (flat layout, not `src/`).

### Environment and conventions (match Brandon's existing `tavily-research-agent` project)
- Python ≥ 3.12, managed with **uv** (`pyproject.toml` + `uv.lock`). `requirements.txt` is
  still required by the rubric: generate it from uv (`uv export --no-hashes --format requirements-txt`)
  and keep it in sync; setup must work with either `uv sync` or `pip install -r requirements.txt`.
- pytest (+ pytest-asyncio if needed) as dev dependencies.
- Secrets via 1Password at runtime: a committed `.env.op` containing `op://` references, run as
  `op run --env-file=.env.op -- <cmd>`. Brandon's key is the 1Password item
  **"Anthropic Academy - API Key"** in vault **Private** (verify the field name with
  `op item get` before writing the reference). LangSmith key: `op://Private/LangSmith API key/credential`.
- `.env.example` documents plain env vars for everyone else (`ANTHROPIC_API_KEY=`, optional
  `LANGSMITH_*`, `CODER_MODEL`, etc.). Code loads `.env` if present (python-dotenv) so non-1Password
  users can copy `.env.example` → `.env`.
- Never commit `.env`, Chroma data, or the agent workspace.
- Local path: `/Volumes/Tukar/Code/ai-certifications/SarasAI-Masters-AI-Eng/multi-agent-dev-team`.
  Creating the public GitHub repo and pushing are outward-facing: confirm with Brandon at that step.

### Model
- Default model: the current Claude Sonnet (`claude-sonnet-5-5` at time of writing) via
  `langchain-anthropic`. Overridable with `CODER_MODEL` env var / `--model` flag.
- Check current LangGraph / langchain-anthropic / chromadb APIs against live docs (context7 or the
  LangChain docs MCP) when implementing; do not rely on memory for API shapes.

## 6. How it works (architecture)

### 6.1 Graph (hand-built LangGraph `StateGraph`, not the prebuilt ReAct agent)
Hand-built so the rubric-critical behaviours are explicit and inspectable: memory retrieval before
every LLM call, the iteration counter, and the validated finaliser.

```
START → retrieve_memory → agent ──(action calls, or text-only nudge, n < limit)──→ tools → retrieve_memory → agent …
                            │
                            finish alone (any n) / limit reached without finish
                            ↓
                         finalize → END        (full precedence rules below)
```

- **State:** task, messages, iteration count, plan, list of tool events (action + observation),
  files written, last execution result, retrieved memories, status.
- **retrieve_memory:** runs before every `agent` call. Queries Chroma with the task plus the latest
  observation; injects retrieved items and the sliding-window buffer into the system/context
  message for that call. Records what was retrieved in state (so CLI/UI/tests can show it).
- **agent:** Claude Sonnet with tools bound. Each turn is one ReAct iteration; it emits brief
  reasoning text (the "Thought" shown to the user — hidden model reasoning is not required) and
  tool calls. Increments the iteration counter. First turn must produce a numbered plan, stored in
  state.
- **tools:** executes the tool calls **sequentially, in the order the model emitted them** (do not
  use a parallel tool executor such as the prebuilt `ToolNode` default). If a call *errors*
  (rejected path, invalid arguments, tool exception), the remaining calls in that batch are not run
  and each gets a "skipped: an earlier call in this batch failed" observation (every tool call still
  receives a result message, as the Anthropic API requires). A non-zero exit code or timeout from
  `exec_python` is an observation, not a tool error, so later calls still run. Observations go back
  into messages so the next action is chosen from observed results. Tested: `write_file` then
  `exec_python` in one response runs the freshly written code; an error mid-batch skips the rest.
- **Iteration limit:** `MAX_ITERATIONS = 10` (configurable down, never above 10). LangGraph's
  own `recursion_limit` is set as a backstop, but the explicit counter is the mechanism.
- **Routing after each `agent` call** (precedence, top wins; *n* = calls made so far):
  1. `finish` is the **only** tool call → validate its args → `finalize`. Applies at *n* = limit too:
     a valid finish on the last call is honoured. Invalid args → error observation, treated as case 4.
  2. `finish` **mixed** with other tool calls → `finish` is rejected with an observation ("finish
     must be called alone, after your last action"); the other calls are handled as case 3.
  3. Ordinary tool calls → if *n* < limit, execute them → `retrieve_memory` → `agent`. If *n* = limit,
     they are **not** executed (the agent could never see their results) → `finalize` with
     `status = "iteration_limit"`.
  4. No tool calls (plain text only) → if *n* < limit, add a nudge observation ("call a tool, or
     call finish") and loop; if *n* = limit → `finalize` with `status = "iteration_limit"`.
  Tests cover: finish exactly at the limit (success), ordinary tools at the limit (not executed,
  iteration_limit), mixed batch, text-only reply, all with a scripted fake model.
- **finalize:** builds the Pydantic output (see 6.4). It does not ask the LLM to restate the code;
  `code` comes from the files on disk and `result` from the last real execution.

### 6.2 Tools (`tools/`)
**Path confinement (scope of the guarantee):** every *path argument* to the file tools and every
*execution target* of `exec_python` is resolved (`Path.resolve()`, so symlinks and `..` are
followed) and rejected unless it lies inside the shared **workspace directory** (`./workspace/`,
gitignored; see §9.2). This is all the confinement Module 1 provides. **Code executed by
`exec_python` runs as a normal host process and keeps full access to the host filesystem and
network**; `cwd` and interpreter flags are not isolation. The README's "Known limitations" says
this plainly; real isolation is the Module 4 Docker work.

- `write_file(path, content)` → writes, returns bytes/lines written.
- `read_file(path)` → returns content (truncated with a marker if large).
- `exec_python(path | code, timeout=None)` — execution contract:
  - Interpreter: the project's own interpreter (`sys.executable`), so the agent's code sees the
    same Python and installed packages (including pytest).
  - `cwd = workspace`. A `.py` script runs as `sys.executable <file>` **without `-I`/`-P`**, so the
    script's directory is on `sys.path` and sibling imports (`from palindrome import ...`) work.
    `-s` (no user site-packages) is fine. Never run with `-O` (asserts must execute).
  - A test file (`test_*.py` / `*_test.py`, or an explicit `mode="pytest"`) runs as
    `sys.executable -m pytest -q <file>` from the workspace, so pytest's default import mode puts
    the test's directory on `sys.path` and assertions actually run. Exit code is pytest's.
  - Returns stdout/stderr, exit code, duration, `timed_out`, plus the target path and an
    **execution record**: the set of workspace files that were actually loaded during the run
    (the target itself plus every module imported from the workspace), each with its SHA-256 at
    load time. Collected by a small runner module the subprocess starts with
    (`sys.executable -m <runner> script|pytest <target>`): it runs the target (via `runpy` for
    scripts with the target's directory on `sys.path`, or `pytest.main` for tests) and, at exit,
    writes the workspace files present in `sys.modules` (plus the target) to a JSON sidecar. The
    runner is a separate process, so this is still not in-process `eval`/`exec` in the agent.
    Used by §6.4 verification.
  - Default timeout 10 s, configurable per call and via env (`EXEC_TIMEOUT_SECONDS`).
  - **Process-tree cleanup:** launch with `subprocess.Popen`, not `subprocess.run` (`run` only kills
    the direct child on timeout). POSIX (macOS, Linux, Docker): `start_new_session=True`, and on
    timeout `os.killpg(pgid, SIGKILL)`, then reap and drain pipes. Windows:
    `CREATE_NEW_PROCESS_GROUP` and `taskkill /T /F /PID`. Clear timeout observation returned.
  - Never `eval`/`exec` in-process.
- Inline code passed to `exec_python` is written to a file in the workspace first, so every
  executed artifact exists on disk.
- Tests: an infinite loop **and** a program that spawns a long-running child process are both
  stopped within ~timeout with no surviving descendants; a two-file acceptance test
  (`palindrome.py` + `test_palindrome.py` importing it) passes under pytest mode; a failing
  assertion yields a non-zero exit code; symlink and `..` escapes are rejected.

**Concurrency:** one active run per workspace, enforced across processes with a lock file in the
workspace parent (e.g. the `filelock` package, non-blocking acquire). A second run (CLI or web UI)
gets a clear "workspace busy" error instead of queuing; the web UI disables Run while a run is
active. `--reset-memory` takes the same lock and refuses while a run holds it. Tested with two
overlapping acquisitions.

### 6.3 Memory (`memory/` package or inside `agents/` — builder's call)
- **Sliding window:** last N conversation turns (a turn = user task + agent final answer; see §9.4),
  N configurable (`MEMORY_WINDOW_TURNS`, default 10). Persists across tasks within a CLI session /
  UI server process.
- **Reset generation:** a small file next to the lock holds a reset counter. `--reset-memory`
  (under the workspace lock) clears Chroma and the workspace and increments the counter. Every
  process remembers the generation its buffer belongs to and, at the start of each run (after
  taking the lock), clears its in-process buffer if the counter has moved. So a reset from the CLI
  also empties an idle web UI's buffer before its next run. Tested: process A holds turns, process
  B resets, A's next run sees an empty buffer.
- **Semantic store:** Chroma `PersistentClient` at `./.chroma/` (gitignored), one collection.
  Embeddings: Chroma's default local embedding function (no extra API key; Anthropic has no
  embeddings endpoint). After each run, store task, plan, file names, a short code summary, and
  outcome. Retrieval: top-k (default 3) by similarity, with a distance threshold so unrelated
  junk isn't injected.
- **Demonstration:** the demo's second task ("now make it ignore punctuation") must show the first
  task's work being retrieved and used. Also cross-session: a fresh process recalls prior runs.
- A `--reset-memory` flag / helper clears both layers for clean demos and tests.

### 6.4 Structured output (Pydantic)
`CoderResult` (validated with Pydantic v2):
- `code: str` — contents of the primary file(s) written, read back from disk (multi-file:
  concatenated with filename headers, plus a `files` mapping).
- `explanation: str` — from the agent's `finish` tool call (or templated from state if the
  iteration limit was hit; see §9.3).
- `plan: list[str]` — the plan the agent used (from its first turn, updated if it revised).
- `result: ExecutionResult` — from the last real `exec_python` call: exit code, stdout, stderr,
  timed_out, duration. If code was never executed, the result says so and `status` cannot be
  `success`.
- Extra fields: `status` (`success | failed | iteration_limit | error`), `iterations`, `files`,
  `memories_used`, `run_id`.
- `verified: bool` — true only if **every delivered file was actually executed, in its current
  form, by a successful run**. Delivered files = the `.py` files this run wrote. For each, there
  must be a successful (exit 0, not timed out) `exec_python` after its last write whose execution
  record (§6.2) shows that file was loaded (as the target or as an imported module) with the
  same SHA-256 as on disk now. Hash freshness alone is not enough; the file must have been part
  of what ran. Any delivered file that fails this makes `verified = false`, and the result names
  which file and why ("not executed", "changed after last successful run").
- `finish` also takes an optional `entry_point` (the script or test file that demonstrates the
  deliverable); if given, it must be the target of a qualifying successful run.
- A model validator enforces consistency: `status == success` requires a non-timeout, exit-code-0
  result **and** `verified == true`. A run that wrote version A, ran it successfully, then overwrote
  it with B and finished is reported `failed` ("code changed after last successful run"), never
  success. Everything factual comes from state; `finalize` makes no model call.
- Regression tests (scripted fake model):
  - write A → exec (pass) → write B → finish ⇒ not success, `verified` false.
  - write broken `palindrome.py` → write `ok.py` (`print("ok")`, imports nothing) → exec `ok.py`
    (pass) → finish ⇒ not success, `verified` false (`palindrome.py` "not executed").
  - write `palindrome.py` + `test_palindrome.py` → exec the test (pass) → finish ⇒ success,
    both files verified via the execution record.

### 6.5 Entry points
- `agents/coder_agent.py` exposes a `CoderAgent` (or `run_coder(task, ...)`) returning
  `CoderResult`, plus a streaming interface yielding step events (used by CLI and UI).
- **CLI** (`uv run coder "task"`): live step output using `rich` (memory → thought → action →
  observation, with "n/10"), then a result panel. Flags: `--model`, `--max-iterations`,
  `--timeout`, `--reset-memory`, `--json` (print the CoderResult JSON), `--demo`.
- **`--demo` / `make demo`:** runs the two related tasks back-to-back (palindrome checker with
  tests → "now make it ignore punctuation"), showing memory recall on the second.
- **Web UI** (`uv run coder-web`, binds localhost): FastAPI (or similar) server streaming step
  events over SSE/WebSocket to a single static page (plain HTML/JS; a CDN syntax highlighter is
  fine; no React build step). Layout B as in §4. Read-only. Same agent code path as the CLI.
- Optional LangSmith tracing when `LANGSMITH_TRACING=true` and a key is present; never required.

### 6.6 Tests (`tests/`)
- **Unit (no API key needed):** tools (workspace confinement, read/write, exec success, exec
  timeout kills an infinite loop within ~timeout, no `eval` path), sliding window eviction,
  Chroma store/retrieve round-trip, `CoderResult` validation rules, iteration limit enforced
  using a fake/scripted chat model that always requests another tool call.
- **Smoke / integration (marked `live`):** invokes the real Coder, using whatever provider
  `CODER_MODEL` selects, with a short task; asserts every `CoderResult` field is present and
  valid, that `code` equals the file on disk, that `verified` is true, and status is success. Also
  a memory follow-up test. **Gating follows the configured provider's prerequisites**:
  `anthropic:` needs `ANTHROPIC_API_KEY`, `openai:` needs `OPENAI_API_KEY`, `ollama:` needs a
  reachable Ollama server with the model pulled. The skip reason names the missing prerequisite.
- **Skips never look like passes:** the documented verification command (`make verify`, also in
  the README and the rubric-evidence table) runs pytest with `-rs` and `REQUIRE_LIVE=1`, which turns
  any skipped `live` test into a failure. Plain `make test` / `uv run pytest` runs everything and
  may skip `live` tests. README shows how to run with `op run`.

### 6.7 Placeholders and deployment files
- `agents/pm_agent.py`, `agents/qa_agent.py`: docstring stubs describing their future role.
- `Dockerfile` + `docker-compose.yml`: actually work for the Coder CLI and web UI (key passed via
  env), even though Docker is formally Module 4.
- `docs/`: architecture diagram (Mermaid in markdown is fine) and a usage guide.

### 6.8 README (portfolio + assessor)
Top: one-paragraph pitch + GIF/screenshot of the web UI (added once it exists). Then: quickstart
(1Password path and plain `.env` path), CLI and web usage, architecture diagram, implementation
decisions, **"Rubric evidence" table** mapping C1.1–C1.6 to files and to a command that
demonstrates each (including the timeout and iteration-limit probes), known limitations
(subprocess is not a security sandbox), roadmap for Modules 2–5.

## 7. Order of work
1. Repo scaffold + README stub + initial commit (course says the structure must exist from the
   first commit).
2. Tools → memory → graph → structured output → CLI → tests, until C1.1–C1.6 are all demonstrably met.
3. Web UI.
4. Docker, docs, README polish, demo GIF.
5. Create public GitHub repo and push (confirm with Brandon first); note the commit SHA for submission.

## 8. Open items / flags for Brandon
- The LMS shows the Module 1 due date as "1 April 2026", which looks wrong for an Aug 2026 cohort;
  confirm with course staff.
- The setup guide describes a private SarasAI-provided repo; the project guide requires a public
  one. We're going public per Brandon; worth confirming with staff.

## 9. Calls made after builder review (Claude's calls; Brandon to check)

1. **Assessor may not have an Anthropic key.** The model is created with LangChain's
   `init_chat_model` from a provider-prefixed string (`CODER_MODEL`, default
   `anthropic:claude-sonnet-5-5`), so `openai:gpt-4o` or `ollama:<model>` work by swapping one env
   var (the course-allowed alternatives). `langchain-openai` and `langchain-ollama` are optional
   extras; README documents each. In addition, a real run's evidence is committed
   (`docs/evidence/`: CLI transcript + `CoderResult` JSON + smoke-test output at the submitted
   commit, and a LangSmith trace link if available). The smoke test stays a *real* model call; no
   fake model in the smoke test.
2. **Workspace and follow-up tasks.** One persistent workspace per project directory
   (`./workspace/`, gitignored), shared by all runs and sessions, cleared by `--reset-memory`
   (which clears buffer, Chroma, and workspace together). Runs are not isolated from each other;
   that's what lets the follow-up task `read_file` and modify `palindrome.py`. Chroma metadata
   stores the file paths each run wrote, so retrieved memories point at real files the agent can
   read. Tool path arguments and execution targets stay confined to `./workspace/` (see §6.2 for
   exactly what that does and doesn't cover); `--reset-memory` and runs share the workspace lock.
   The web UI file tree shows the whole workspace, highlighting files touched in the current run.
3. **What counts as an iteration, and no hidden 11th call.** One iteration = one `agent` model
   call (one Thought → Action → Observation cycle). The agent finishes by calling a `finish` tool
   with its `explanation` and final `plan`; `finalize` makes **no** model call. If the limit is hit,
   `finalize` builds the explanation from state (template: what was attempted, last observation,
   why it stopped). So every model call is an `agent` call, every one is preceded by
   `retrieve_memory`, and the total is ≤ 10. The CLI/UI counter shows model calls.
4. **The sliding window is cross-task conversation memory, not an in-run trimmer.** It holds the
   last N conversation turns (a turn = user task + agent's final answer/summary), default N = 10,
   and is injected into the context message alongside Chroma results. The current run's ReAct
   messages are never trimmed (bounded by the 10-iteration cap), so tool calls and results are
   never split. The eviction unit test targets the buffer directly.

## 10. Left to the builder
Exact module/file names within the required dirs, prompt wording, Chroma collection schema,
SSE vs WebSocket, CSS/visual polish of the UI (dark theme, matching the mockup), `Makefile`
targets, exact `exec_python` signature (path vs inline code vs pytest mode), and logging format.
