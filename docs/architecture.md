# Architecture

The Coder agent is a hand-built LangGraph `StateGraph`, not the prebuilt ReAct agent. The rubric-critical behaviour stays explicit and testable: memory retrieval before every model call, the iteration counter, sequential tool execution, and a finaliser that builds the validated result without asking the model.

## The graph

```mermaid
flowchart LR
    START((start)) --> RM[retrieve_memory]
    RM --> A[agent<br/>one model call = one iteration]
    A -- "ordinary tool calls, n &lt; limit<br/>or text only (nudge)" --> T[tools<br/>sequential, in emitted order]
    T --> RM
    A -- "finish alone (any n)<br/>or limit reached" --> F[finalize<br/>no model call]
    F --> END((end))
```

| Node | Code | What it does |
|---|---|---|
| `retrieve_memory` | `orchestration/graph.py` | Queries Chroma with the task plus the latest observation, formats the hits and the sliding window into a `# Memory` block, and emits a `memory` event. It runs before **every** `agent` call. |
| `agent` | `orchestration/graph.py` | Sends `SystemMessage(SYSTEM_PROMPT + memory)` plus the run's messages to the model. It increments `iterations`, extracts a numbered plan, and classifies the reply (see routing). |
| `tools` | `orchestration/graph.py`, `tools/dispatch.py` | Answers rejected calls, runs the remaining calls one by one, and appends a `ToolMessage` for each. The first `ToolError` skips the rest of the batch. A non-zero exit code or a timeout is an observation, not an error. |
| `finalize` | `orchestration/finalize.py` | Builds `CoderResult` from state and from the files on disk, then runs delivery verification. It makes no model call. |

The system message is rebuilt for each call and never stored in state. Memory therefore reaches every call, and the stored conversation stays exactly what the model said and observed.

## Routing precedence

`orchestration/routing.py::classify`. *n* is the number of model calls so far, including this one. The top matching rule wins.

1. **`finish` is the only call.** Its arguments are validated, then the run finalizes. This also applies at *n* = limit. Invalid arguments are rejected with an observation and treated as rule 4.
2. **`finish` mixed with other calls.** `finish` is rejected ("finish must be called alone, after your last action") and the other calls fall through to rule 3.
3. **Ordinary calls.** They execute if *n* < limit. At the limit they are **not** executed, because the agent could never see their results. Status becomes `iteration_limit`.
4. **No tool calls.** The agent is nudged to call a tool or finish if *n* < limit; otherwise status becomes `iteration_limit`.

The limit is `CODER_MAX_ITERATIONS` (default and hard cap: 10). `Settings` rejects anything outside 1–10. LangGraph's `recursion_limit` (3 × limit + 5) is only a backstop.

## Tools and the execution record

- **Confinement** (`tools/workspace.py`). Every file-tool path and every execution target is resolved, following `..` and symlinks, and must lie inside `./workspace/`. A leading `workspace/` prefix is stripped, because models often add one.
- **`exec_python`** (`tools/exec_python.py`). It runs `sys.executable -s tools/_runner.py <mode> <target> …` with `cwd=workspace`, `stdin=/dev/null`, and its own process group.
  - When the run ends, the **whole process group** is killed with `SIGKILL`, including any children the script spawned. On timeout (default 10 s) that happens early.
  - Bytecode is cached outside the workspace, and the runner drops the workspace's cached modules before each run. A same-size rewrite within one second can't run stale `.pyc` code.
  - Output is drained by reader threads that keep the first 8,000 bytes per stream. A print flood can't deadlock the run or fill memory or disk.
  - Test files (`test_*.py`, `*_test.py`) run under pytest. Inline code is written to `.coder_snippets/` first, so every executed artifact exists on disk.
- **The runner** (`tools/_runner.py`) runs inside the child process. It installs a `MetaPathFinder` that records the SHA-256 of every workspace module at the moment it is imported, plus the target itself. At exit it also sweeps `sys.modules` for workspace files loaded by other importers, such as `conftest.py` loaded by pytest's assertion rewriter. It writes the result to a JSON record.
- **Files written by executed code.** `.py` files that a run creates or changes, rather than `write_file`, count as writes made at that step. They are reported in the observation, delivered, and need a later successful run to be verified.
- **Verification** (`orchestration/verify.py`). A delivered `.py` file is verified only if some successful run after its last write loaded it with the hash it has on disk now. So "wrote A, ran it, overwrote with B, finished" is reported `failed`, and so is a broken module that was never imported by the passing run. `CoderResult`'s model validator then refuses `status="success"` unless the last run passed and delivery is verified.

## Memory

- **Sliding window** (`memory/window.py`). The last N conversation turns, where a turn is a task plus the agent's final explanation (`MEMORY_WINDOW_TURNS`, default 10). It lives in the process. The current run's ReAct messages are never trimmed; the 10-call cap bounds them.
- **Semantic store** (`memory/semantic.py`). A Chroma `PersistentClient` at `./.chroma/`, using cosine distance and Chroma's default local embedding (all-MiniLM-L6-v2 via ONNX, no API key).
  - After each run the agent stores its task, outcome, files, plan, summary, and the names of the top-level functions and classes it delivered.
  - Retrieval returns the top 3 hits with distance ≤ 0.65.
- **Reset** (`CoderAgent.reset_memory`). Under the workspace lock, it clears Chroma, the workspace, and the window, and bumps a generation counter in `.coder/generation`. Every process checks the counter at the start of each run and drops its stale window, so a reset from the CLI also reaches an idle web server.

## Concurrency

There is one active run per workspace across processes, enforced by `filelock` on `.coder/workspace.lock` with a non-blocking acquire. A second run gets "workspace busy": exit code 2 in the CLI, HTTP 409 in the web UI. The holder also writes its pid to `.coder/running`. The web UI's status poll reads that marker instead of probing the lock, so polling can never make a starting run see "busy". The Run button stays disabled while any live process holds the lock.

## Streaming

`CoderAgent.stream(task)` yields `StepEvent`s (`start`, `memory`, `thought`, `plan`, `action`, `observation`, `nudge`, `limit`, `error`, `result`). Graph nodes emit them with `get_stream_writer()`, and the agent reads them with `stream_mode=["custom", "values"]`. The CLI renders them with rich. The web server forwards them as server-sent events on `POST /api/run`. A dedicated thread drives each run. When the browser disconnects, that thread stops at the next event and closes the run, which releases the lock.

## v2: the PM → Coder team graph

Module 2 wraps the Coder in an outer graph (`orchestration/team_graph.py`, built by `build_team_graph(TeamDeps(pm, coder))`). The inner graph above is untouched and runs inside `coder_node`. The diagram is in the [README](../README.md#v20-pm--coder), and `tests/test_docs.py` checks it against the compiled graph.

| Node | What it does |
|---|---|
| `pm_node` | `PMAgent.draft_spec(requirement)` returns a validated `Spec`. On a `PMError` it records `error` and `status="failed"`. |
| `consolidate_node` | Runs only when the PM returned more than `MAX_TASKS` (8) tasks. Merges them (model first, deterministic fallback) and records a `Consolidation`. |
| `dispatch_node` | Builds a `TaskHandoff` for `tasks[current_task_index]`, sets `handoff`, marks the task `in_progress`. |
| `coder_node` | Reads only `state["handoff"]`, renders it with `render_task`, runs one `CoderAgent` for the whole project, forwards each step event, records the task result. |
| `finalize` | Sets the run status: `completed` (every task `done`), `partial` (some `failed`), `failed` (PM error or no tasks). |

Routers: `route_after_pm` sends an error or an empty task list to `finalize`, more than 8 tasks to `consolidate_node`, otherwise to `dispatch_node`. `route_after_coder` sends the run back to `dispatch_node` while tasks remain, otherwise to `finalize`. A failed task does not stop the run.

### `ProjectState`

Defined in `orchestration/project_state.py`. Fields without a reducer are overwritten by the node that owns them.

| Field | Type | Owner (writes it) | Reducer |
|---|---|---|---|
| `requirement` | `str` | set at entry | none, never modified |
| `spec` | `Spec \| None` | `pm_node` | none (written once) |
| `tasks` | `list[Task]` | `pm_node`; `consolidate_node` replaces it | none (overwrite) |
| `consolidation` | `Consolidation \| None` | `consolidate_node` | none |
| `task_status` | `dict[str, TaskRecord]` | `pm_node` (pending), `consolidate_node` (pending), `dispatch_node` (`in_progress`), `coder_node` (`done` / `failed`) | `merge_dicts` (later keys win) |
| `current_task_index` | `int` | `coder_node` (+1 per task) | none |
| `handoff` | `TaskHandoff \| None` | `dispatch_node` sets it, `coder_node` clears it | none |
| `artifacts` | `list[TaskArtifact]` | `coder_node` (one `CoderResult` per task) | `operator.add` |
| `code_output` | `dict[str, str]` | `coder_node` (path to content) | `merge_dicts` (later tasks win) |
| `work_in_progress` | `list[str]` | every node (chronological log) | `operator.add` |
| `error` | `str \| None` | `pm_node` | none |
| `status` | `"running" \| "completed" \| "partial" \| "failed"` | `pm_node` on failure, `finalize` | none |

### Handoff protocol

`dispatch_node` is the readiness signal: it writes a `TaskHandoff` into state before the Coder starts, so the handoff is visible in the streamed state. `coder_node` then uses that object and nothing else. It never sees `spec`, `tasks` or `requirement`. `render_task(handoff)` turns it into the Coder's task text: project name and description, "Task T2 (2 of 4, high priority)", the title and description, the acceptance criteria as a checklist, the files earlier tasks produced, and the instruction to build on them, keep earlier tests passing and run all tests before finishing.

Every run writes `handoff-<task_id>.json` into `.coder/runs/<run id>/`. An example of what `handoff-T1.json` looks like (illustrative values; the real file from the demo run will be in `docs/evidence/v2/`):

```json
{
  "task_id": "T1",
  "title": "Create the FastAPI app with GET /health",
  "description": "Create main.py with a FastAPI app and a GET /health endpoint that returns {\"status\": \"ok\"}.",
  "acceptance_criteria": [
    "GET /health returns HTTP 200 with the JSON body {\"status\": \"ok\"}",
    "main.py defines a module-level `app = FastAPI(...)`"
  ],
  "priority": "high",
  "position": 1,
  "total": 2,
  "spec_name": "Echo API",
  "spec_description": "A small REST API with a health check and an echo endpoint.",
  "prior_files": []
}
```

Later tasks get `prior_files` filled from the keys of `code_output` (for T2: `["main.py", "test_main.py"]`, say).

Locking and limits: `TeamRunner` holds the workspace lock for the whole project run, so no other `coder` or `team` run can interleave between tasks. Each task keeps v1's limits: at most 10 Coder iterations and a 10 s exec timeout.

## Extension points for Modules 2–5

- **Module 2 (PM agent), built.** `agents/pm_agent.py` produces a spec; the outer graph (see above) hands each task to the Coder as a `TaskHandoff`, and the structured `CoderResult` is the return contract.
- **Module 3 (QA agent).** `CoderResult.files`, `result` and `verification_notes` give QA everything it needs to re-run and critique. A QA loop becomes a second graph that calls the Coder again with feedback.
- **Module 4 (MCP/A2A, Docker).** The tools are plain functions behind `tools/dispatch.py::run_action`, so they can be exposed over MCP unchanged. Real isolation replaces the subprocess with a container per run.
- **Module 5 (observability, cost).** LangSmith tracing already works through environment variables. `StepEvent` is the place to add token and cost accounting.
