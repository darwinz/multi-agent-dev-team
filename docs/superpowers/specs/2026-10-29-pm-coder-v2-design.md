# Multi-Agent System v2.0 (PM → Coder) — Design

**Course:** Saras AI Institute, *Build Autonomous Multi-Agent Systems* (Aug 2026), Module 2 course project
**Date:** 2026-10-29
**Builds on:** [Coder Agent v1.0 design](2026-10-25-coder-agent-v1-design.md) (merged in PRs #1 and #2)
**Source of requirements:** the Module 2 Course Project Guide and the six-criterion grading rubric (C2.1–C2.6).
Brandon opted to skip the requirements interview; every choice not traceable to those two documents is
marked *(my call)* and collected in §14.

## 1. Intent

Module 1 delivered one agent: a bounded ReAct Coder that writes, runs and verifies Python in a
workspace. Module 2 puts a **Product Manager agent** in front of it. The PM turns a free-text
requirement into a structured specification and task list; the Coder works through the tasks in
order; both read and write one typed shared state inside one LangGraph graph.

Two goals, as in Module 1:

1. **Meet all six rubric criteria cleanly**, including the probes graders run: an input that makes
   the PM produce more than eight tasks, and a Postman-style request fixture run against the API the
   system builds for the published demo task.
2. **Stay portfolio-worthy**: one command shows a requirement becoming a spec, then tasks being built
   and verified one by one, with evidence a reviewer can inspect afterwards.

## 2. Source requirements

### Project guide, Module 2 instructions (numbered as in the guide)
8. PM agent: parse a free-text requirement, generate a structured technical specification (name,
   description, acceptance criteria, list of tasks), write it into shared state.
9. PM → Coder handoff: the Coder receives the task list from shared state, executes each task in
   order, and reports progress back to shared state after each task completes.
10. Shared TypedDict (or equivalent) state that both agents read and write, containing: the original
    requirement, the technical spec, the task list, per-task status, accumulated code output.
11. Handoff protocol — a structured message or state key the Coder reads to know a task is ready —
    carrying task ID, description, acceptance criteria and priority.
12. Loop prevention: if the PM assigns more than 8 tasks for one requirement, it consolidates before
    proceeding.
13. Tests cover the handoff: after the PM node runs, shared state holds a non-empty task list with the
    correct schema.
14. README system diagram showing both agents and the data flow between them (ASCII, or PNG in `docs/`).

Technical requirements: both agents in the same orchestration graph (no separate processes); state
schema in Pydantic or TypedDict, no ad-hoc dictionaries; handoff serializable to JSON; framework
continues from Module 1 (LangGraph) so no justification is needed.

Deliverables: `agents/pm_agent.py`; updated `orchestration/` wiring both agents; the shared state
schema; the handoff protocol; the 8-task limit; the integration test; the README diagram. A
cross-framework comparison doc (Lab 2.4 findings) is recommended, not required.

**Demo task used in grading:** "Build a REST API with two endpoints: GET /health (returns {status: ok})
and POST /echo (returns the JSON body it receives). Include basic input validation." Graders verify
the spec has at least two tasks, the Coder produces working Flask/FastAPI code, and that code passes a
supplied Postman-style request fixture.

### Rubric details that shape the design
| | Met requires | Not Yet traps |
|---|---|---|
| C2.1 | Spec in state with name, description, acceptance criteria, tasks covering the requested behaviour; ≥2 tasks for the API task | Prose-only spec; missing fields; lost behaviour |
| C2.2 | Both agents read/update the same schema with all five fields; updates keep context later tasks need | Ad-hoc dict; missing field; an update that discards context |
| C2.3 | Same graph; a JSON-serializable message/key signals readiness with ID, description, criteria, priority; **the Coder consumes those fields** | Manual copying; no readiness signal; fields not consumed |
| C2.4 | In-order execution; progress reported **after each task**; code output accumulates; >8 tasks consolidated to ≤8 **keeping requirement coverage** | Status changes only at the end; skipped tasks; limit ignored; deleting extra tasks |
| C2.5 | Automated test runs the **real PM node** and checks the task schema; the demo task runs through PM and Coder and the API passes the fixture (/health, /echo, invalid input) | Tests that bypass the PM; unchecked schema; a failing endpoint |
| C2.6 | README diagram shows PM, Coder, shared state and data flow, **matching the code**; instructions that work | Diagram that misrepresents the graph |

Graders look for "state captured after the PM runs", "before-and-after state from an observed
handoff", "serialized handoff ... linked to the same task ID", and "consolidated task list and
requirement mapping". The design produces each of these as a file (§9).

## 3. Goals, non-goals, anti-goals

**Goals:** the PM agent; the outer PM → Coder graph; the shared `ProjectState`; the handoff message;
coverage-preserving consolidation; per-task progress in state and on screen; a `team` CLI; the demo
API passing a fixture we ship; tests (unit, integration, live); README diagram and docs.

**Non-goals for this module:**
- Web UI changes. `coder-web` keeps working for single Coder runs; a team view can come later.
- The QA agent, MCP, A2A (Module 3) and container isolation (Module 4).
- Parallel task execution, task dependencies beyond list order, LangGraph checkpoint persistence.
- Rebuilding the PM in AutoGen/CrewAI inside this repo. The comparison doc stays optional (§12).

**Anti-goals** (would fail the module even if the code runs):
- A Coder that ignores the handoff and works from the raw requirement.
- Consolidation that drops tasks or acceptance criteria.
- A grading demo that only passes because of leftovers from an earlier run in the workspace.
- A README diagram that drifts from the real graph.

## 4. Architecture

```
            requirement
                │
   START ──► pm_node ──route_after_pm──┬── error ──────────────────────────► finalize ──► END
                                       ├── > 8 tasks ──► consolidate_node ──┐      ▲
                                       └── ≤ 8 tasks ───────────────────────┤      │
                                                                            ▼      │
                                         ┌──────────────────────────► dispatch_node│
                                         │                                  │ handoff (TaskHandoff)
                                         │                                  ▼      │
                                         └──── more tasks ◄── route_after_coder ◄── coder_node
                                                               └── all done ───────┘
```

- **One graph** (`orchestration/team_graph.py`), state `ProjectState`. Node and router names follow
  Lab 2.3 (`pm_node`, `coder_node`, `route_after_pm`, `route_after_coder`) so graders recognise them.
- `consolidate_node` and `dispatch_node` are additions *(my call)*. Consolidation needs its own
  observable step for C2.4. Dispatch makes the readiness signal an explicit state write (`handoff`)
  that appears in the streamed state before the Coder starts, which is the C2.3 evidence.
- `coder_node` calls the existing `CoderAgent` in-process for one task. The v1 inner ReAct graph
  (`orchestration/graph.py`) is untouched and runs inside that node. No separate processes.
- One `CoderAgent` instance serves every task of a project run, so its sliding-window and semantic
  memory carry context from task to task, and every task shares one workspace.

## 5. Shared state and data models

New module `orchestration/project_state.py`. All models are Pydantic v2 with `extra="forbid"`; state
is a `TypedDict` whose fields hold those models. Everything round-trips through `model_dump_json()`.

```python
Priority = Literal["high", "medium", "low"]
TaskStatus = Literal["pending", "in_progress", "done", "failed"]

class AcceptanceCriterion(BaseModel):
    id: str            # "AC1", "AC2", ... unique within a spec
    text: str          # testable statement of required behaviour

class Task(BaseModel):
    task_id: str                     # "T1", "T2", ... unique within the list
    title: str
    description: str
    acceptance_criteria: list[str]   # task-level, min 1, each checkable by running code
    priority: Priority
    covers: list[str]                # ids of spec-level AcceptanceCriteria this task satisfies, min 1

class Spec(BaseModel):
    name: str
    description: str
    acceptance_criteria: list[AcceptanceCriterion]   # min 1
    tasks: list[Task]                                 # min 1; the PM's original decomposition
    # validators: unique ids; every `covers` id exists; every spec criterion is covered by ≥1 task

class TaskHandoff(BaseModel):          # the readiness signal (C2.3)
    task_id: str
    title: str
    description: str
    acceptance_criteria: list[str]
    priority: Priority
    position: int                      # 1-based index in the execution list
    total: int
    spec_name: str
    spec_description: str
    prior_files: list[str]             # workspace files produced by earlier tasks

class TaskRecord(BaseModel):           # per-task progress (C2.4)
    task_id: str
    status: TaskStatus
    coder_status: Status | None = None   # the CoderResult status (success / failed / iteration_limit / error)
    run_id: str | None = None
    iterations: int = 0
    files: list[str] = []
    summary: str = ""                    # CoderResult.explanation, truncated

class Consolidation(BaseModel):        # requirement mapping evidence (C2.4)
    original_tasks: list[Task]
    merged_from: dict[str, list[str]]  # new task_id → original task_ids, a partition of the originals
    method: Literal["llm", "deterministic"]

class ProjectState(TypedDict):
    requirement: str                                         # set at entry, never modified
    spec: Spec | None                                        # written once by pm_node
    tasks: list[Task]                                        # execution list (≤ 8); = spec.tasks unless consolidated
    consolidation: Consolidation | None
    task_status: Annotated[dict[str, TaskRecord], merge_dicts]
    current_task_index: int
    handoff: TaskHandoff | None                              # set by dispatch_node, cleared by coder_node
    artifacts: Annotated[list[TaskArtifact], operator.add]   # one per finished task: record + CoderResult dump
    code_output: Annotated[dict[str, str], merge_dicts]      # accumulated path → content, later tasks win
    work_in_progress: Annotated[list[str], operator.add]     # chronological progress log (Lab 2.3 name)
    error: str | None
    status: Literal["running", "completed", "partial", "failed"]
```

`TaskArtifact` is `{task_id: str, result: CoderResult}`. The five required fields map to `requirement`,
`spec`, `tasks`, `task_status` and `code_output`. No node replaces a field it doesn't own; reducers
append or merge, so later tasks never lose earlier context (C2.2).

`spec.tasks` keeps the PM's original decomposition even after consolidation, so the requirement
mapping is always inspectable next to the execution list.

## 6. PM agent (`agents/pm_agent.py`)

- `class PMAgent` with `draft_spec(requirement) -> Spec` and `consolidate(spec, max_tasks) -> (list[Task], Consolidation)`.
  `pm_node` and `consolidate_node` are thin wrappers that call it and return state updates.
- Model: the same `init_chat_model` setting as the Coder (`CODER_MODEL`, default
  `anthropic:claude-sonnet-5-5`), injectable for tests. Output via `with_structured_output(Spec)`
  (tool-calling) — no JSON parsing of prose.
- Prompt (`agents/prompts.py`, `PM_SYSTEM_PROMPT`) tells the PM to:
  - restate the requirement as numbered, testable spec-level acceptance criteria, including every
    stated behaviour and implied validation/error cases;
  - decompose into **one task per independently testable unit of behaviour**, in build order, each
    small enough for a Coder run of at most 10 turns and each ending with its tests passing;
  - give each task task-level acceptance criteria, a priority, and the spec criteria it covers;
  - for web APIs, specify FastAPI, an ASGI `app` object, and tests that use `fastapi.testclient`.
  The prompt does **not** mention the 8-task limit *(my call)*. Consolidation is a separate,
  observable step; telling the PM about the cap would hide it and leave the grader's ">8 tasks" probe
  untestable.
- Validation: if the structured output fails Pydantic validation (including the coverage
  validator), retry once with the validation errors appended. A second failure sets `error`, status
  `failed`, and routes to `finalize` without running the Coder.

## 7. Consolidation (> 8 tasks)

`MAX_TASKS = 8` (constant in `project_state.py`). `route_after_pm` sends the run to
`consolidate_node` when `len(spec.tasks) > MAX_TASKS`.

1. **LLM merge.** The PM is asked to merge the tasks into at most 8, combining adjacent or related
   ones, returning new tasks plus `merged_from`. The result is accepted only if:
   - there are ≤ 8 tasks;
   - `merged_from` is a partition of the original task ids (each appears exactly once);
   - every spec-level acceptance criterion is still covered (union of `covers`);
   - each merged task's `covers` includes the `covers` of all its sources;
   - order respects the original order of the first source of each group.
2. **Deterministic fallback** if the LLM result fails validation or the call errors: split the
   original list, in order, into 8 consecutive groups as evenly as possible; each merged task takes
   the joined titles, concatenated descriptions, concatenated acceptance criteria, the highest
   priority, and the union of `covers`. This always satisfies the checks above. `method` records
   which path ran.

Nothing is deleted: every original task and criterion ends up inside a merged task, and
`consolidation` stores both lists and the mapping.

## 8. Handoff and the Coder node

- `dispatch_node`: takes `tasks[current_task_index]`, builds a `TaskHandoff` (with `prior_files` =
  keys of `code_output`), sets `handoff`, marks that task `in_progress`, appends to `work_in_progress`.
- `coder_node`:
  1. reads **only** `state["handoff"]` (plus nothing else from the PM's fields) and renders the Coder
     task text from it with `render_task(handoff) -> str`: project name and description, "Task T2 of 4
     (high priority)", title, description, the acceptance criteria as a checklist, the files earlier
     tasks produced, and the instruction to build on them, keep earlier tests passing, and run all
     tests before finishing;
  2. runs `CoderAgent.stream(task_text)` and forwards each inner `StepEvent` through LangGraph's
     stream writer, tagged with the task id;
  3. returns: `task_status[id]` = `done` (CoderResult `success`) or `failed` (anything else), with
     run id, iterations, files and summary; one `TaskArtifact`; `code_output` merged with the
     result's files; `current_task_index + 1`; `handoff = None`; a progress line.
- `route_after_coder`: next task if any remain, else `finalize`.
- **A failed task does not stop the run** *(my call)*: it is recorded as `failed` and the next task
  still runs (no skipping). `finalize` sets status `completed` if every task is `done`, `partial` if
  some failed, `failed` if the PM failed.
- The workspace lock (`RunLock`) must be held for the whole project run, not just per task, so
  another process can't interleave between tasks. If `RunLock` is not re-entrant, `CoderAgent.stream`
  gets a way to run under a lock the caller already holds.
- Each task keeps v1's limits: at most 10 Coder iterations, 10 s exec timeout.

## 9. Running it: CLI, progress and evidence

New console script `team` (`app/team_cli.py`, registered in `pyproject.toml`):

```
team "<requirement>"            run PM → Coder on a requirement
team --demo api                 the published grading task (§10)
team --demo large               a requirement that reliably yields > 8 tasks (exercises consolidation)
team ... --reset-memory         clear memory and workspace first (recommended for the demos)
team ... --json                 print the final ProjectState as JSON on stdout
```

`--model`, `--max-iterations` and `--timeout` pass through as in `coder`. Exit code 0 only when the
run status is `completed`.

Terminal rendering (rich), in order: the spec panel (name, description, criteria, tasks with
priority and covers); the consolidation panel when it ran (original → merged mapping); for each task
a header ("T2/4 · high · Add POST /echo"), the Coder's step events indented beneath it, then a
one-line status; a final table of task id, status, iterations and files.

**Evidence files.** Every run writes to `.coder/runs/<project_run_id>/`:
`requirement.txt`, `state-after-pm.json`, `consolidation.json` (when it ran), `handoff-<task_id>.json`
for each task, `state-after-<task_id>.json` after each task, and `final-state.json`. The run prints
this path. A captured demo run is copied into `docs/evidence/` for the submission, as in Module 1.

## 10. The demo API and the request fixture

- The generated API is FastAPI *(my call)*: it is already a dependency, and Flask isn't installed.
  `httpx` moves from the dev group to main dependencies so code in the workspace can use
  `fastapi.testclient` when the Coder runs its tests.
- Expected behaviour, encoded in our fixture and in the demo's acceptance criteria:
  `GET /health` → 200 `{"status": "ok"}`; `POST /echo` with a JSON object → 200, the same object;
  `POST /echo` with malformed JSON, a non-object body or no body → 422 (FastAPI's validation status).
- **Fixture:** `tests/fixtures/demo_api.postman_collection.json`, a Postman v2.1 collection. Each
  request carries saved example responses (`response[]` with `code` and `body`), the standard Postman
  way to record expected results. The same file can be run with `newman` against a live server.
- **Fixture runner:** `scripts/check_api.py --app <module>:app [--collection PATH]` starts the
  generated app with uvicorn on a free local port, sends each request over real HTTP, and compares
  status codes and JSON bodies with the saved examples. Without `--app` it finds the module in the
  workspace that defines a FastAPI `app`. Output is a pass/fail line per request; exit 0 if all pass.
- `team --demo api` runs the runner automatically after the last task and prints its results.

## 11. Tests

Unit and integration tests use the scripted fake chat model from `tests/fakes.py`, extended so
`with_structured_output` returns scripted `Spec` objects. No network, as in v1.

| File | What it pins |
|---|---|
| `tests/test_project_state.py` | Spec validators (unique ids, unknown `covers` ids, uncovered criterion rejected); `TaskHandoff` and full-state JSON round-trip |
| `tests/test_pm_agent.py` | `pm_node` writes `spec` and `tasks`; one repair retry on invalid output; second failure sets `error` and stops before the Coder |
| `tests/test_consolidation.py` | 11 tasks → ≤ 8; every original id in exactly one group; every criterion still covered; invalid LLM merge → deterministic fallback; ≤ 8 tasks untouched |
| `tests/test_team_graph.py` | **The C2.5 integration test.** Runs the compiled team graph with a fake PM model and asserts, from the streamed state after `pm_node`, a non-empty `tasks` list where every item validates as `Task`. Also: a `handoff` with the first task's id appears before the Coder runs; `render_task` output contains the handoff's id, description, criteria and priority; statuses change after each task, in order; `code_output` accumulates; a failed task is recorded and the next still runs |
| `tests/test_team_coder_real.py` | `coder_node` with the real `CoderAgent` and a scripted model that writes and runs a file: the artifact and `code_output` reflect it |
| `tests/test_check_api.py` | The fixture runner passes against a reference app in `tests/fixtures/reference_api.py` and fails against a deliberately broken one |
| `tests/test_team_cli.py` | Argument handling, `--json` output, evidence files written |
| `tests/live/test_team_demo.py` (`live` marker) | The real demo task: spec has ≥ 2 tasks, run status `completed`, fixture runner passes; the `large` demo consolidates to ≤ 8 with coverage |

## 12. Documentation

- **README:** a "v2.0: PM → Coder" section with the system diagram as Mermaid (renders on GitHub)
  plus the ASCII version from §4; how to run `team` and the fixture; the rubric-to-evidence table
  (§13); a pointer to the evidence files. The diagram is checked against `team_graph.py` node and
  edge names before submission.
- `docs/architecture.md` gains the outer graph, the state table, and the handoff protocol.
- Optional: `docs/framework-comparison.md` summarising Lab 2.4 findings (LangGraph vs AutoGen vs
  CrewAI for the PM role). Recommended by the guide, not graded.
- The personal course notes in the README's LOCAL-NOTES block stay out of every commit.

## 13. Rubric → evidence

| Criterion | Evidence |
|---|---|
| C2.1 | `agents/pm_agent.py`; `requirement.txt` + `state-after-pm.json` from the demo run; spec with ≥ 2 tasks |
| C2.2 | `ProjectState` in `orchestration/project_state.py`; `state-after-pm.json` vs `state-after-T1.json` |
| C2.3 | `orchestration/team_graph.py`; `handoff-T1.json`; `render_task` and its test |
| C2.4 | `state-after-<id>.json` sequence; `work_in_progress` log; `team --demo large` → `consolidation.json` |
| C2.5 | `tests/test_team_graph.py` result; demo run; `scripts/check_api.py` output on the generated API |
| C2.6 | README diagram and instructions; `docs/architecture.md` |

## 14. Decisions made without Brandon's input

1. PM uses tool-calling structured output (`with_structured_output(Spec)`), one repair retry.
2. The PM prompt doesn't mention the 8-task cap; consolidation is its own node.
3. Consolidation: LLM merge validated against coverage, deterministic fallback.
4. Added `dispatch_node` (explicit readiness signal) and `consolidate_node` beyond Lab 2.3's two nodes.
5. `spec.tasks` keeps the original decomposition; `tasks` is the execution list.
6. A failed task is recorded and the run continues; final status `partial`.
7. FastAPI only; invalid input on `/echo` returns 422; `httpx` becomes a main dependency.
8. The fixture is a Postman v2.1 collection with saved example responses, plus our own runner.
9. No web UI changes this module.
10. Evidence files under `.coder/runs/<id>/` on every run.

## 15. Open questions (Brandon / course staff)

1. **Lab repos vs this repo.** Module 2 has four lab repos (`SarasAI-Institute/Multi-agents-lab-2-1`
   to `-2-4`). In Module 1 the labs were completed first in their own forks. Do the Module 2 labs
   need to be done (and submitted) before or alongside this?
2. **Grader fixture format and the invalid-input status.** The rubric says graders supply the
   fixture, but not its format or whether invalid input should return 400 or 422. Our 422 is
   FastAPI's convention; if the grader expects 400, the demo's acceptance criteria change.
3. **Where to submit.** Pending Anshuman's reply about the SarasAI-provided repo; this design
   assumes the public `darwinz/multi-agent-dev-team`.

## 16. Left to the builder

File-internal structure, prompt wording beyond the rules above, rich styling, how the fake model
scripts structured output, the `large` demo's exact wording (it must reliably yield > 8 tasks with
the default model), and the evenly-split arithmetic for the fallback.
