# Multi-Agent System v4.0 (production hardening): Design

Status: draft for Brandon's review, 2026-11-08.
Builds on: v3 (`docs/superpowers/specs/2026-11-02-qa-a2a-v3-design.md`), branch `feat/qa-a2a-v3` at `1ec8029`.
Work happens in the worktree `multi-agent-dev-team-v4`, on branch `feat/production-v4`.
Course sources: the Module 4 Project Guide, the Module 4 submission rubric (C4.1–C4.9), and the Section 1–3 notes. All are summarized in `../multi-agent-module-4-notes/MODULE-4-NOTES.md` (outside this repo), whose "Section 4: Course Project" section holds the full rubric text.

## 1. Intent

Make the existing PM → Coder ↔ QA team production-grade **without adding agents**:
- tracing;
- per-call retry and circuit breakers;
- token and cost accounting;
- caching with a TTL;
- a model per role;
- tighter execution security;
- a test suite that proves each of these;
- a clean Docker Compose deployment that runs the grading task end to end.

Priorities, in Brandon's words: "both, but passing all nine first try matters most". v4 must clear every rubric criterion on the first submission. It must also stay a system he keeps using and showing, so the build quality matters beyond the grade.

The v3 property carries over: a grader reading the evidence (trace, logs, cost report, run directory) can verify each claim without trusting the model's own account.

## 2. Source requirements

### Project guide, Module 4 instructions (numbered as in the guide)
22. Tracing (LangSmith or Phoenix) of all three agents. Each agent invocation is a child span of a single root trace, with metadata `agent_name`, `token_count`, `tool_calls`, `duration_ms`.
23. Retry on all LLM and tool calls: max 3 retries, exponential backoff with jitter. A circuit breaker opens after 3 consecutive failures and half-opens after a configurable cool-down. When the circuit is open, return a cached result or a structured error and never crash.
24. Per-agent token tracking (prompt, completion, estimated USD). Prompt caching with a configurable TTL. A larger model for the PM, smaller models for the Coder and QA.
25. Pydantic validation of every LLM output before acting on it. Subprocess sandbox for code. Secrets live in env vars loaded from `.env`, never in source or images.
26. Tests:
    - mocked unit tests per agent;
    - integration tests for PM+Coder and Coder+QA;
    - at least one E2E test on a realistic task, asserting the output is executable Python that passes a supplied fixture.
27. Dockerfile: Python 3.11, installs `requirements.txt`, sets `ENTRYPOINT`. `docker-compose.yml` has an agent service and a Chroma service, with a health check so the agent waits for Chroma. A clean `docker compose up` runs the pipeline end to end.
28. Each run automatically writes a cost report JSON to `docs/cost_reports/`: per-agent usage, per-agent cost, and the total.

Technical requirements:
- one root trace, not three separate ones;
- the retry wraps individual calls, not the agent loop;
- breaker transitions are logged at WARNING;
- tokens are recorded after every call, not estimated at the end;
- Docker secrets come in via `env_file`;
- 100% pass rate on the mocked unit and integration tests.

Grading demo: "Build a Python module that implements a binary search tree with insert, search, and in-order traversal methods, with full test coverage." Graders will:
- run `docker compose up`;
- check the dashboard for a single root trace;
- check the cost report JSON;
- confirm the mocked suite passes 100%.

### Rubric details that shape the design (9 points, each criterion must be Met)
- **C4.1 (tracing):**
  - one root trace per run, with every PM/Coder/QA invocation as a child span carrying the four fields;
  - structured logs and the dashboard let an assessor follow **the same run**.
- **C4.2 (retry):**
  - every LLM and tool call is retried at the call boundary, ≤ 3 retries counted separately from the original attempt, with jitter;
  - a simulated transient failure gets retried;
  - a persistent failure stops at the bound without replaying the agent loop.
- **C4.3 (breaker):**
  - opens after 3 consecutive failures and blocks calls while open;
  - half-opens after a configurable cool-down;
  - a successful probe closes it and a failed probe reopens it;
  - transitions are logged at WARNING;
  - when open, the call returns a cached result or a structured error, never a crash.
- **C4.4 (cost):**
  - after **every** call, record prompt and completion tokens plus USD per agent;
  - the run writes the JSON automatically;
  - totals reconcile with the recorded calls.
- **C4.5 (cache and models):**
  - a repeated eligible request within the TTL hits the cache, and an expired one doesn't;
  - a larger PM model, smaller Coder/QA models;
  - **no % cost reduction is required.**
- **C4.6 (validation and execution):**
  - external inputs and tool arguments are checked before use;
  - every LLM output passes Pydantic before it's acted on;
  - **an invalid output never reaches tools or execution** (graders probe this);
  - subprocess execution with timeouts;
  - no real secrets in source or images.
- **C4.7 (tests):**
  - mocked unit tests for each agent;
  - mocked integration tests for PM→Coder and for Coder→QA termination, all passing;
  - an E2E test that runs the generated Python against the supplied fixture, including the BST behaviours;
  - "a green suite alone does not establish required coverage."
- **C4.8 (deploy):**
  - Python 3.11, `requirements.txt`, `ENTRYPOINT`;
  - agent and Chroma services, readiness via health check, `env_file`;
  - a clean start (no stale container or host-only artifact) runs the task;
  - evidence: clean-start logs showing Chroma ready before the agent runs.
- **C4.9 (docs):**
  - architecture, setup and config, an API / callable-interface reference, a usage guide;
  - how to run the tests, find traces and cost reports, and read success vs an unresolved-issue exit;
  - everything matches the deployed system.

## 3. Current state (recon of v3, 2026-11-08)

- **Model creation:** one factory, `agents/model.py:make_model` (`init_chat_model`), returning a bare chat model.
- **LLM call sites** (7). Each agent calls `.bind_tools` / `.with_structured_output(..., include_raw=True)` / `.invoke` itself:
  - Coder ReAct step (`orchestration/graph.py:79`);
  - Coder critique (`agents/coder_agent.py:91,100`);
  - PM spec and PM consolidation (`agents/pm_agent.py:29,41,55`);
  - QA suite, fix plan and test audit (`agents/qa_agent.py:165,194,355,388`).
- **Tool calls:**
  - the Coder's go through one dispatcher, `tools/dispatch.py:run_action`;
  - **QA calls `exec_python` / `write_file` directly** (`qa_agent.py:226,240,368`);
  - Chroma reads and writes happen in `memory/semantic.py`.
- **No retry/backoff, logging, token accounting, per-role models, or tracing code.** LangSmith works today only through env vars.
- **Pydantic coverage is already strong:**
  - structured outputs for Spec, MergePlan, QATestSuite, FixPlan, TestAudit, Critique;
  - tool arguments are validated via `ARG_MODELS`;
  - `QAReport.approved` is computed, not taken from the model.
- **`exec_python` passes the full `os.environ` to generated code** (`tools/exec_python.py:131`), so API keys are visible to LLM-written code.
- **Chroma:** a single `PersistentClient` in `memory/semantic.py:21`, with one collection (`coder_runs`, cosine). Only the Coder uses it.
- **Python:** `requires-python >=3.12`, and `.python-version` is 3.12.
- **Docker:**
  - the Dockerfile uses `python:3.12-slim` with `uv sync` and `CMD ["coder-web"]`;
  - Compose has `web` + a `coder` CLI profile, with no team service, no Chroma server and no health check;
  - only `ANTHROPIC_API_KEY` is passed, via `environment:`.
- **Tests:** 346 mocked tests pass in ~7 min.
  - LLMs are mocked with `tests/fakes.py:ScriptedChatModel`.
  - Live tests in `tests/live/` are gated by key presence only, so **a plain `pytest` makes paid calls when a key is set.**
- **Output:** `rich` console rendering plus the evidence files in `.coder/runs/<id>/`.

## 4. Goals, non-goals, anti-goals

**Goals:** every bullet in §2, implemented and evidenced, plus the decisions in §5.

**Non-goals:**
- MCP;
- streamable HTTP, Kubernetes, vaults;
- Phoenix;
- a semantic cache;
- rewriting the mocks to the HTTP level;
- a ≥ 20% cost-reduction study;
- a team mode for the web UI;
- container/network isolation for `exec_python` beyond the env change below.

**Anti-goals** (a failure even if the rubric passes):
- **Regressing v3 behaviour.** The C3.x behaviours must still hold: QA writes and runs real tests, reflection happens before QA, the A2A trust boundary holds, the unresolved-issue report is honest. The existing 346 tests must stay green, unless a change is deliberate and noted in the plan.
- **A plain `pytest` (or `make test`) spending money.**
- **Secrets appearing anywhere they shouldn't:** generated-code environments, the image, logs, the cost report, or trace metadata Brandon would make public.
- **Retries that hide permanent errors,** or that replay a whole agent loop.
- **The cache feeding a stale answer back into a revision loop,** which would make QA retest unchanged code.
- **Docs that disagree with what `docker compose up` actually does.**

## 5. Decisions

Marked **(B)** = Brandon decided in conversation; **(C)** = Claude's call, open to Brandon's review.

### 5.1 Where and how
- **(B)** v4 is the next version of Brandon's own `multi-agent-dev-team` repo, not a fork of the `Multi-Agent-lab-4-1` starter.
- **(C)** Worktree `multi-agent-dev-team-v4`, branch `feat/production-v4` off `feat/qa-a2a-v3`, the same pattern as v3.
- **(C)** Python moves to **3.11** everywhere:
  - `requires-python = ">=3.11"`, `.python-version` set to 3.11, `uv.lock` regenerated, `requirements.txt` re-exported;
  - the builder checks for 3.12-only syntax (PEP 695 `type` aliases or generic syntax, etc.) and fixes it.

### 5.2 Tracing (C4.1)
- **(B) LangSmith** is the tracer. Phoenix is not used.
- **(C) One root trace per team run.**
  - The team graph is invoked with a `run_name` such as `team-run:<run_id>`, plus metadata `run_id` (the same id as `.coder/runs/<run_id>/` and the cost report filename).
  - Every PM, Coder (including reflection and revision) and QA node execution is a child span of that root.
  - The rubric's "agent invocation" means each node execution. Each node span carries `agent_name` (`pm`, `coder`, `qa`), `token_count` (the sum over that invocation's LLM calls), `tool_calls` (a count, plus tool names), and `duration_ms`.
  - These values come from the same per-call records that feed the cost report (§5.5), so trace and report agree.
  - How to attach metadata known only at node end is left to the builder: wrapping nodes with `traceable` and setting run-tree metadata before the span closes, or an equivalent. It must be visible in the LangSmith UI on the node span.
- **(C) Tracing without a key.** If `LANGSMITH_API_KEY` / `LANGSMITH_TRACING` aren't set, tracing is a silent no-op and nothing fails. A grader without Brandon's account still gets a full run plus logs.
- **(C) Structured logs.**
  - JSON lines, one event per line, with `ts`, `level`, `run_id`, `agent`, `event` and event fields.
  - Written to stderr and to `.coder/runs/<run_id>/events.jsonl`.
  - Events include each LLM call (model, tokens, cost, cache hit, duration, attempt), each retry attempt and its delay, breaker transitions (WARNING), tool calls, and run start/end.
  - When tracing is on, the LangSmith trace URL/ID is logged at run start, so logs and dashboard tie to the same run.
  - stdlib `logging` with a JSON formatter versus `structlog` is the builder's choice; prefer no new dependency if it's simple.
- **(B)** Brandon will run the BST demo with tracing on and put a **public LangSmith share link** to that root trace in the README and the evidence folder. It shows prompts and generated code, never keys.

### 5.3 Retry (C4.2)
- **(C) Retry covers everything.** It wraps **every LLM call** (all 7 sites, through the model factory) and **every tool call**:
  - `run_action`;
  - QA's direct `exec_python` / `write_file`;
  - Chroma reads and writes in `memory/semantic.py`.
- **(C) The retry budget.**
  - At most **3 retries** after the original attempt (4 attempts total), set by `RETRY_MAX` (default 3, hard cap 3).
  - Exponential backoff with **full jitter**: `uniform(0, min(RETRY_MAX_DELAY_S, RETRY_BASE_DELAY_S * 2**attempt))`. Defaults: base 1 s, max 30 s.
  - Tests set the delays to ~0.
- **(C) Which errors are retried.**
  - Only **transient** errors: rate limit (429), timeouts, 5xx/overloaded (incl. Anthropic 529), connection errors, and transient OS errors on tools.
  - Permanent errors (auth, 4xx validation, model not found) fail straight away.
  - A program's own non-zero exit or timeout inside `exec_python` is a **result, not a tool failure**, and is never retried.
- **(C) Retried writes must be idempotent.**
  - Today `SemanticMemory.add()` makes a fresh `uuid4` id inside the call (`memory/semantic.py:33`). If Chroma commits an insert, the response is lost and the call is retried, a duplicate memory entry would appear.
  - The id is therefore fixed **once, before the retry loop**: generated by the caller or derived from a hash of the text + metadata. Every attempt then writes the same id with `upsert`, so a replayed write is a no-op.
  - The same rule covers every retried write: `write_file` already overwrites the same path, so it's idempotent. Any new retried write must be idempotent or deduplicated before it gets a retry.
  - A test injects a "committed but response lost" failure and asserts there's exactly one entry.
- **(C) Provider SDK retries go to 0** (e.g. `max_retries=0` on the Anthropic client). Our wrapper is then the only retry layer, and the count is honest.
- **(C) Retry sits at the call boundary.** It must work with `bind_tools` and `with_structured_output(..., include_raw=True)` exactly as the agents use them today. The mechanism is the builder's choice: a delegating chat-model wrapper, a custom Runnable, or an equivalent. Known trap: LangChain's `.with_retry()` returns a Runnable that no longer has `bind_tools`.
- **(C) Logging.** Each attempt is logged with attempt number, error class and chosen delay. Exhausting the retries logs `retry_exhausted` and raises a typed `TransientError` / `PermanentError` (a small exception hierarchy as in the Section 2 notes). Existing agent handling then turns that into `PMError`, the Coder `error` result, or QA's `qa_error`, as today.

### 5.4 Circuit breakers (C4.3)
- **(C) Breaker scope.** One breaker per agent role (`pm`, `coder`, `qa`) for LLM calls, plus one for `tools` and one for `chroma`. Each is shared by every call of that kind in the process.
  - Sharing across a role's different operations (e.g. PM spec and PM consolidation) is deliberate. Each role maps to exactly one model per run (§5.6), so the role breaker *is* the breaker for that model endpoint. Transient failures on any of its operations are evidence the endpoint is unhealthy.
- **(C) Order and counting.** The breaker sits **outside** the retry.
  - **Only transient failures count:** a call that ended in `TransientError` after exhausting its retries (rate limit, timeout, 5xx/overloaded, connection error, transient OS error).
  - **These never count and never open a breaker**, because they don't indicate an outage; they go through the existing agent error handling as today:
    - permanent provider errors (auth, 4xx validation, model not found);
    - structured-output parsing or Pydantic validation failures;
    - `exec_python` program results.
  - A **success resets** the consecutive-failure count. A permanent error neither increments nor resets it.
- **(C) State behaviour.**
  - **CLOSED:** opens after `BREAKER_FAILURE_THRESHOLD` (default 3) consecutive failures.
  - **OPEN:** rejects calls without trying them.
  - **HALF-OPEN:** after `BREAKER_COOLDOWN_S` (default 30), exactly **one** probe is allowed, and concurrent callers are rejected while it runs. Success closes the breaker and a failure reopens it.
  - Every transition is logged at **WARNING** with breaker name, from/to state and failure count.
- **(C) Open-circuit handling.**
  - If the call is cache-eligible (§5.6) and an **unexpired** cache entry exists, return it and log `degraded_fallback_cached`.
  - Otherwise raise a structured `CircuitOpenError` (a `TransientError` subtype) that the existing paths turn into a clean, reported failure: `PMError` → a failed run with a final state written, Coder `error`, QA `qa_error` → the unresolved-issue report.
  - **The process never crashes on an open circuit.**

### 5.5 Token and cost accounting (C4.4)
- **(C) After every LLM call**, record from `usage_metadata`:
  - role, model, prompt (input) tokens, completion (output) tokens, cache-read and cache-write tokens (from `input_token_details`);
  - whether it was a response-cache hit (0 tokens, $0);
  - estimated USD, duration and attempt count.
- **(C) Where records go.** They live in a per-run collector and are logged as JSON events.
- **(C) Pricing.**
  - Prices come from one table in config, per model: input, output, cache read, cache write, per million tokens.
  - The builder fills them from current Anthropic pricing via the `claude-api` skill, **not from memory**, and notes the date.
  - An unknown model logs a warning and is costed at $0 (flagged in the report), rather than crashing.
- **(C) The report.**
  - At the end of **every** run (success, unresolved, failed, or aborted), `TeamRunner` writes `docs/cost_reports/<run_id>.json` automatically.
  - Contents:
    - `run_id`, `started_at`, `finished_at`, `requirement`, `outcome`;
    - `models` by role, the `pricing` used, `trace_url` (if tracing is on);
    - `calls`: the full per-call list;
    - `agents.{pm,coder,qa}`: calls, prompt/completion/cache tokens, cache hits, `cost_usd`;
    - `total` with the same fields.
  - Totals are computed from `calls`, so they reconcile by construction, and a test asserts this.
  - **Written even on unexpected aborts.**
    - Today `TeamRunner.stream()` writes `final-state.json` only after `graph.stream()` finishes (`orchestration/team_runner.py:68-110`). An exception from graph build, streaming, a node escaping its error handling, or a post-run check skips that write.
    - v4 wraps everything after the run directory is created in `try/except/finally`:
      - on an escaping exception, record `outcome: "aborted"` with the error class and message;
      - write `final-state.json` (whatever state was last seen) and the cost report from the calls recorded so far, in `finally`;
      - log `run_aborted` at ERROR;
      - then re-raise, so the CLI still exits non-zero.
    - A generator closed early by its consumer (`GeneratorExit`) also takes the `finally` path, with `outcome: "aborted"`.
    - A busy workspace (`RunLock` refused) makes no LLM calls and creates no run, so it writes no report. This is unchanged.
    - Tests cover an exception raised inside a node, during streaming, and in a post-run check. Each asserts that a report exists, matches the recorded calls and shows the aborted outcome.
  - A copy goes in the run's evidence dir.
- **(C) Path in Docker.** `docs/cost_reports/` is bind-mounted, so reports from a container run land on the host.

### 5.6 Caching (C4.5)
- **(C) Two layers.**
  1. **Response cache with a configurable TTL.**
     - Keyed by SHA-256 of (model, messages, bound tools/schema, relevant params).
     - Stored on disk under `.coder/cache/`, so it survives across runs and a second demo run within the TTL shows hits.
     - Setting: `CACHE_TTL_S` (default 3600; 0 disables).
     - Lookups log `cache_hit` / `cache_miss` / `cache_expired`.
  2. **Anthropic prompt caching** (`cache_control`) on the long, stable system prompts and tool schemas, to cut real cost. Cache-read/write tokens are recorded in §5.5.
- **(C) Eligibility.** Only calls with **no side effects whose answer shouldn't change within a run** use the response cache:
  - the PM spec and PM consolidation;
  - QA test-suite generation, fix plan and test audit.
  - **Not** the Coder's ReAct steps or critique/revision. A revision can never replay a stale answer, and QA always retests new code.
  - Eligibility is declared per call site, not guessed.
- **(B) Models per role.**
  - PM defaults to Sonnet (`anthropic:claude-sonnet-5-5`).
  - Coder and QA default to Haiku (`anthropic:claude-haiku-5-5`).
  - Env overrides: `PM_MODEL`, `CODER_MODEL`, `QA_MODEL`.
  - These defaults are used for the graded run. Day to day, Brandon switches the Coder to Sonnet via `CODER_MODEL`.
  - The existing single `CODER_MODEL` setting changes meaning to "Coder role only". This is documented as a v4 change.
  - `make_model` takes the role.

### 5.7 Validation and execution security (C4.6)
- **(C) Generated code gets no secrets.** `exec_python` builds a **minimal allowlisted environment** for generated code (`PATH`, `HOME`, `LANG`/`LC_*`, `PYTHONPATH` / whatever the runner needs, `TMPDIR`) instead of `os.environ`. No API keys reach generated code. A test asserts that a key in the parent env is absent in the child.
- **(C) Invalid-output probe.**
  - A test plus a scripted controlled run feed a deliberately invalid structured output (and an invalid tool call) and show that nothing reaches `run_action`, `write_file` or `exec_python`, and that the failure is reported.
  - The builder audits the recon's list of LLM text acted on without semantic validation:
    - `extract_plan` regex;
    - free-text fix strings interpolated into prompts;
    - QA's spliced `TestVerdict.corrected` code, which only gets AST + `.check()`;
    - silent-fallback `except Exception` in PM consolidation and QA fix-plan/audit.
  - For each, either show it's validated before it can reach a tool or execution, or add validation. Silent fallbacks must at least log a WARNING event.
- **(C) Tool arguments** stay validated at `run_action`. QA's direct tool calls get the same argument models or equivalent checks.
- **(C) Secrets.**
  - Only from env: `.env` via Compose `env_file` for graders; `.env.op` + `op run` for Brandon (§5.9).
  - `.dockerignore` excludes `.env`, `.coder/`, `.chroma/`, `workspace/` and `docs/cost_reports/`.
  - The cost report and trace metadata never include env values.

### 5.8 Tests (C4.7)
- **(C) Mocked by default.**
  - Keep `ScriptedChatModel`.
  - `pytest` defaults exclude live tests (`addopts` adds `-m "not live"`).
  - `make test` runs the mocked suite; `make verify` (with `REQUIRE_LIVE=1`) runs the live ones.
- **(C) New mocked unit tests:**
  - **retry:** transient error retried and succeeds; persistent error stops after exactly 3 retries; permanent error not retried; jitter within bounds; agent loop not replayed (call counts);
  - **breaker:** CLOSED→OPEN at 3, calls blocked while OPEN, HALF-OPEN after cool-down with an injected clock, single probe, probe success→CLOSED, probe failure→OPEN, WARNING logs, open-circuit cached fallback and structured error;
  - **cache:** hit within TTL, miss after expiry (injected clock), ineligible calls never cached;
  - **cost:** per-call records, per-agent sums, totals reconcile, report written automatically on success, unresolved and failure;
  - **tracing:** root/child structure and the four metadata fields, against a fake/mocked LangSmith client or run-tree capture; tracing off = no-op;
  - **security:** env allowlist; the invalid-output probe;
  - **models:** role → model resolution and overrides.
- **(C) Unit tests per agent and the two integration paths already exist.** PM: `test_pm_agent.py`. Coder: `test_coder_agent.py` and related. QA: `test_qa_agent.py`. PM→Coder: `test_team_graph.py`. Coder→QA termination: `test_team_loop.py`. The builder makes sure each is clearly identifiable to a grader (names or a README map) and still passes on 3.11.
- **(C) A live E2E test** in `tests/live/`:
  - runs the full team on the BST demo requirement;
  - imports the generated module and runs a **supplied BST fixture** (`tests/fixtures/bst_fixture.py` or similar) against it, covering insert, search (hit and miss) and in-order traversal (sorted output, duplicates policy as the fixture states);
  - asserts it passes, i.e. it doesn't just check text.
  - A `team --demo bst` demo plus a `demo_check` mirrors this for Docker runs.
- **(B) BST interface contract.**
  - The `bst` demo, both E2E tests and `demo_check` send the **published task sentence verbatim, followed by an appended interface block**. The README documents that block as "the supplied fixture's contract".
  - Interface (exact wording is the builder's, semantics fixed):
    - module `bst.py` at the workspace root;
    - class `BinarySearchTree` with `insert(value) -> None`, `search(value) -> bool` and `inorder() -> list` (ascending);
    - duplicate inserts are ignored;
    - values are mutually comparable.
  - The fixture imports `bst.BinarySearchTree` and checks:
    - insert/search hit and miss;
    - in-order traversal sorted, on random and on degenerate (sorted-input) sequences;
    - duplicates ignored;
    - an empty tree.
  - No v3 agent prompt or schema changes for this.
- **(C) Deterministic full-pipeline E2E in the default suite.** Graders without an API key must be able to see C4.7's E2E requirement pass in the default `pytest` run, so the live test is not enough on its own.
  - A mocked E2E test drives the **real** team graph, tools, `exec_python`, QA's pytest run and the BST fixture.
  - Only the LLM is scripted: canned PM spec, Coder tool calls that write working BST code, and QA suite/fix responses. It builds on `tests/scenarios.py`.
  - It asserts the generated module executes and passes the supplied fixture.
  - The live test above stays as a second, real-model proof.
- **(C) What "full test coverage" means in the checks.**
  - **Required:** behavioural coverage of every published method (insert; search hit and miss; in-order traversal), via QA's tests and the supplied fixture.
  - **Reported, not gated:** line coverage of the generated module under QA's tests. `pytest-cov` is added as a dependency, and `demo_check` reports the percentage in its output and evidence.
  - The QA loop is **not** changed to chase a coverage number. That would alter v3 QA behaviour the anti-goals protect.
- **(C) Controlled runs.** Scripted controlled runs (like v3's `scripts/controlled_runs.py`) produce committed evidence for: transient retry then success; persistent failure stopping at the bound; breaker open → half-open → closed and → reopened; cache hit then expiry; the invalid-output probe.

### 5.9 Docker (C4.8)
- **(C) Dockerfile.**
  - `python:3.11-slim`;
  - installs from `requirements.txt` (`pip install -r requirements.txt`, or `uv pip install -r requirements.txt`; the literal step must be visible);
  - keeps the non-root user and the pre-baked embedding model;
  - sets `ENTRYPOINT ["team"]` with a default `CMD` that runs the BST demo.
- **(C) docker-compose.yml.**
  - **`chroma` service:** the official Chroma server image, pinned to a version compatible with the pinned `chromadb` client. It has a named data volume and a `healthcheck` against its heartbeat endpoint; the builder verifies the image has a tool for the check, or uses a Python one-liner.
  - **`team` service (default):**
    - `depends_on: chroma: condition: service_healthy`;
    - `env_file: .env`;
    - bind mounts for `workspace/`, `.coder/` and `docs/cost_reports/`;
    - runs the BST demo once and exits.
  - Chroma keeps running after the team service exits. **(B)** This matches what Brandon wants.
  - The single-Coder web UI and the `coder` CLI move behind profiles (`web`, `cli`).
- **(C) Chroma client.** `memory/semantic.py` uses `chromadb.HttpClient` when `CHROMA_HOST` is set (port via `CHROMA_PORT`), else the existing `PersistentClient` (tests and local runs unchanged).
  - The `reset_memory` cross-process sync assumes a shared `.coder`. That holds in Compose because `.coder` is bind-mounted; note it.
- **(C) Secrets in Compose.**
  - The grader path is a plain `.env` copied from `.env.example`.
  - Brandon's path must work **without writing plaintext keys to disk**: `op run --env-file=.env.op -- docker compose up`, with variables passed through.
  - The builder picks a Compose configuration that supports both and verifies each: e.g. `env_file` with `required: false` plus name-only `environment:` passthrough, checking precedence so an unset host var doesn't blank a value from `.env`.
  - `.env.example` lists every key with placeholder values: Anthropic, LangSmith, model per role, retry/breaker/cache settings, `CHROMA_*`.
- **(C) Clean-start evidence.** `docker compose down -v` + rebuild + `docker compose up`, with logs committed showing Chroma healthy before the team starts and the run's outcome.

### 5.10 Docs (C4.9)
- **(C) README v4 section:**
  - an updated three-agent diagram showing the resilience layer, tracing and the cost report (the `test_docs.py` diagram check must still pass);
  - quickstart for Docker (grader) and local (`op run`);
  - configuration table;
  - **callable-interface reference:** the CLI commands/flags, `TeamRunner`, `make_model`, settings, and the cost report schema;
  - how to run the tests (mocked vs live);
  - where the traces are (public link) and where the cost reports and logs are;
  - **how to read the outcome:** success vs unresolved-issue exit vs failed run.
- **(C) A C4.1–C4.9 evidence map** with file and evidence links, like v3's.
- **(C) Architecture doc.** `docs/architecture.md` gets the resilience/observability layer. `docs/usage.md` gets the new settings and commands.

## 6. Evidence to commit (`docs/evidence/v4/`)
- **Live BST run** (Brandon runs it with his keys):
  - final state;
  - `events.jsonl`;
  - the cost report;
  - public trace link;
  - `demo_check` output.
- **Clean-start Docker logs.**
- **Controlled runs** (scripted, no keys): retry, breaker, cache, invalid-output.
- **Mocked test summary,** 100% pass.

## 7. Left to the builder
Things a sensible builder can settle without changing what gets built:
- module and file layout for the resilience layer (e.g. `agents/resilience.py`, `agents/costs.py`, `observability/`);
- the exact retry-wrapper mechanism (within §5.3's constraints);
- the exact tracing-metadata mechanism (within §5.2);
- the logging library choice;
- cache storage format;
- the exact pinned Chroma image tag;
- the healthcheck command;
- the Compose passthrough syntax;
- test names;
- the task-breakdown order.
