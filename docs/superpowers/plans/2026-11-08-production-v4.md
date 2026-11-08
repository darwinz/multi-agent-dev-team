# Production-Ready System v4.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the v3 PM → Coder ↔ QA team for production: one-root-trace tracing, call-level retry and circuit breakers, per-call token/cost accounting with an automatic cost report, a TTL response cache plus prompt caching, a model per role, a secret-free sandbox, a test suite that proves each, and a Docker Compose deployment that runs the BST grading task.

**Architecture:** Two new packages hold the cross-cutting layer. `resilience/` has the error hierarchy, retry, breakers and response cache. `observability/` has JSON logs, the per-run context, costs and tracing. They plug in at three call boundaries:
- **LLM calls:** a delegating `ResilientChatModel` that `make_model` returns, and that the agents wrap around injected models.
- **Tools:** `run_action` and QA's direct `exec_python`/`write_file` calls.
- **Chroma:** `SemanticMemory`.

`TeamRunner` owns the per-run context, the root trace and the cost report. It writes the report in a `finally`, so aborted runs get one too.

**Tech Stack:** Python 3.11, LangGraph, LangChain (`langchain-anthropic`), LangSmith, Chroma (server in Compose, embedded locally), Pydantic v2, pytest + pytest-cov, uv, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-11-08-production-v4-design.md` (approved 2026-11-08). Read it alongside this plan. Section refs like "§5.3" point there.

## Global Constraints

- **Python:** `requires-python = ">=3.11"`, `.python-version` = `3.11`, Docker base `python:3.11-slim`.
- **Retry:** at most **3 retries** after the original attempt. `RETRY_MAX` defaults to 3 with a hard cap of 3. Full jitter: `uniform(0, min(RETRY_MAX_DELAY_S, RETRY_BASE_DELAY_S * 2**attempt))`, where `attempt` is the 0-based retry index. Defaults: base `1.0` s, max `30.0` s.
- **What gets retried:** only transient errors (429, timeouts, 5xx, overloaded/529, connection errors, transient OS errors). Permanent errors fail straight away. An `exec_python` program's own exit code or timeout is a result, never a failure.
- **Provider SDK retries:** `max_retries=0` (Anthropic/OpenAI clients), so our wrapper is the only retry layer.
- **Breakers:** one each for `pm`, `coder`, `qa` (LLM) plus `tools` and `chroma`.
  - `BREAKER_FAILURE_THRESHOLD` default 3, `BREAKER_COOLDOWN_S` default 30.
  - Only `TransientError` after exhausted retries counts. A success resets the count; a permanent error neither increments nor resets it.
  - Every transition is logged at WARNING.
- **Response cache:** `CACHE_TTL_S` default 3600, where 0 disables it. On disk under `.coder/cache/`. Eligible calls only: PM spec, PM consolidation, QA suite, QA fix plan, QA audit.
- **Models:** `PM_MODEL` = `anthropic:claude-sonnet-5-5`; `CODER_MODEL` = `QA_MODEL` = `anthropic:claude-haiku-5-5`.
- **Cost report:** written to `docs/cost_reports/<run_id>.json`, plus a copy at `.coder/runs/<run_id>/cost-report.json`.
  - Written on every outcome: `success`, `unresolved`, `failed`, `aborted`.
  - Totals are computed from `calls`.
- **Logs:** JSON lines with `ts`, `level`, `run_id`, `agent`, `event` and the event's fields. They go to stderr and to `.coder/runs/<run_id>/events.jsonl`.
- **No secrets** in generated-code environments, the image, logs, the cost report or trace metadata.
- **Default `pytest` and `make test` never make paid calls:** `addopts` includes `-m "not live"`.
- **Existing tests stay green.** Every deliberate test change is named in its task's commit message.
- **New dependencies:** `langsmith` (made explicit) and `pytest-cov` only. Logging uses stdlib `logging`.

## Review Focus

These failure modes are implied by the spec but no section names them. Each line gets a test in the task that owns the code.

1. **Ctrl-C mid-run** (`KeyboardInterrupt` inside a node) still leaves a cost report with `outcome: "aborted"` and `final-state.json`. `team` exits 130 with no traceback. Covered in Task 11.
2. **A corrupt or half-written cache file**, e.g. from a crash or two processes at once, is treated as a miss and overwritten, never a crash. Writes are atomic (`os.replace`). Covered in Task 7.
3. **Chroma unreachable at startup** (`CHROMA_HOST` set, nothing listening): `team` prints `team: cannot run: …Chroma…` and exits 2, with no traceback. Covered in Task 9.
4. **A provider error message that echoes a secret** (an API key value) is redacted before it reaches `events.jsonl`, stderr or the cost report's `error`. Covered in Task 5.
5. **`KeyboardInterrupt` during a retry sleep** propagates at once and is never counted as a retryable failure. Covered in Task 3.

---

## File structure

**New**
- `resilience/__init__.py`: empty.
- `resilience/errors.py`: `AgentError`, `TransientError`, `PermanentError`, `CircuitOpenError`, plus the classifiers.
- `resilience/retry.py`: `RetryPolicy`, `call_with_retry`.
- `resilience/breaker.py`: `BreakerState`, `CircuitBreaker`, `BreakerRegistry`.
- `resilience/runtime.py`: the process-wide `Resilience` (policy + breakers + sleep), `configure`, `runtime`, `reset`, `resilient_call`.
- `resilience/cache.py`: `ResponseCache`.
- `observability/__init__.py`: empty.
- `observability/logs.py`: JSON formatter, `configure_logging`, `log_event`, `redact`, `run_log_file`.
- `observability/context.py`: `RunContext`, `run_scope`, `current_run`, `agent_scope`, `current_agent`, `record_call`, `record_tool`.
- `observability/costs.py`: `Price`, `PRICING`, `PRICING_AS_OF`, `CallRecord`, `ToolRecord`, `cost_usd`, `usage_from_message`, `build_cost_report`, `write_cost_report`.
- `observability/tracing.py`: `tracing_enabled`, `root_trace`, `agent_node`, `span_metadata`.
- `tools/guarded.py`: `guarded_tool`.
- `app/demo_check.py`: add `check_bst` to the existing module.
- `tests/fixtures/bst_fixture.py`: the supplied fixture.
- Tests:
  - `tests/test_retry.py`
  - `tests/test_breaker.py`
  - `tests/test_cache.py`
  - `tests/test_logs.py`
  - `tests/test_costs.py`
  - `tests/test_resilient_model.py`
  - `tests/test_tracing.py`
  - `tests/test_security.py`
  - `tests/test_e2e_bst.py`
- `scripts/resilience_runs.py`: v4 controlled-run evidence.
- `docs/evidence/v4/README.md`: the C4.1–C4.9 evidence map.

**Modified**
- Packaging and environment: `pyproject.toml`, `.python-version`, `uv.lock`, `requirements.txt`, `Makefile`.
- Docker and secrets: `Dockerfile`, `docker-compose.yml`, `.env.example`, `.env.op`, `.dockerignore`.
- Config and model factory: `agents/config.py`, `agents/model.py`.
- Agents: `agents/pm_agent.py`, `agents/coder_agent.py`, `agents/qa_agent.py`.
- Tools and memory: `tools/dispatch.py`, `tools/exec_python.py`, `memory/semantic.py`.
- Orchestration: `orchestration/team_graph.py`, `orchestration/team_runner.py`.
- App: `app/team_cli.py`.
- Tests: `tests/conftest.py`, `tests/live/conftest.py`, `tests/live/test_team_demo.py`, `tests/scenarios.py`, `tests/test_config.py`.
- Docs: `README.md`, `docs/architecture.md`, `docs/usage.md`.

---

### Task 1: Python 3.11, dependencies, mocked-by-default tests

**Files:**
- Modify: `pyproject.toml`, `.python-version`, `uv.lock`, `requirements.txt`, `Makefile`, `tests/conftest.py`, `tests/live/conftest.py`
- Test: existing suite

**Interfaces:**
- Produces: the `[tool.pytest.ini_options] addopts = "-p no:cacheprovider -m 'not live'"` setting.
- Produces: an autouse fixture `_fast_resilience` in `tests/conftest.py`. It sets these env vars for every non-live test: `RETRY_BASE_DELAY_S=0`, `RETRY_MAX_DELAY_S=0`, `CACHE_TTL_S=0`. Once Task 3 exists it also calls `resilience.runtime.reset()`.

- [ ] **Step 1: Switch to 3.11.**
  - Set `.python-version` to `3.11` and `requires-python = ">=3.11"`.
  - Add `"langsmith>=0.14.12"` and `"pytest-cov>=6"` to `dependencies`.
  - Add `"resilience"` and `"observability"` to `[tool.hatch.build.targets.wheel] packages`.
  - Run `uv python install 3.11 && uv lock && uv sync`.
- [ ] **Step 2: Scan for 3.12-only syntax.**
  - Look for PEP 695 `type X = …` / `def f[T]`, `typing.override`, `itertools.batched`, and PEP 701 same-quote nested f-strings.
  - Run: `uv run python -m compileall -q agents tools memory orchestration app scripts tests`.
  - Expected: no `SyntaxError`. Fix any hits in place.
- [ ] **Step 3: Exclude live tests by default.**
  - Set `addopts = "-p no:cacheprovider -m 'not live'"`.
  - Makefile:
    - `verify: REQUIRE_LIVE=1 uv run pytest -m live -rs`. The command-line `-m` overrides `addopts`.
    - Add `team-bst: uv run team --demo bst --reset-memory` (it works once Task 14 lands).
- [ ] **Step 4: Add the env fixtures.**
  - Add the autouse `_fast_resilience` fixture (monkeypatch env as above) to `tests/conftest.py`.
  - In `tests/live/conftest.py`, add an autouse fixture that `monkeypatch.delenv`s those three vars, so live runs use real delays and the cache defaults.
- [ ] **Step 5: Run the full mocked suite on 3.11.**
  - Run: `uv run python --version && uv run pytest -q`.
  - Expected: `Python 3.11.x`, then 346 passed, 0 failed, with live tests deselected.
- [ ] **Step 6: Export requirements and commit.**
  - Run `make requirements`. `requirements.txt` must now list `langsmith` and `pytest-cov`.
  - Commit `chore: move to Python 3.11, add langsmith and pytest-cov, exclude live tests by default`.

### Task 2: Settings: model per role and resilience settings

**Files:**
- Modify: `agents/config.py`, `agents/model.py` (only `provider_prerequisite` callers here), `app/team_cli.py`, `tests/live/conftest.py`, `.env.example`, `.env.op`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces `Role = Literal["pm", "coder", "qa"]` in `agents/config.py`.
- Produces these new `Settings` fields and env vars:

  | Field | Type and default | Env var |
  |---|---|---|
  | `model` (now the Coder role only) | default `"anthropic:claude-haiku-5-5"` | `CODER_MODEL` |
  | `pm_model` | `str = "anthropic:claude-sonnet-5-5"` | `PM_MODEL` |
  | `qa_model` | `str = "anthropic:claude-haiku-5-5"` | `QA_MODEL` |
  | `retry_max` | `int = Field(3, ge=0, le=3)` | `RETRY_MAX` |
  | `retry_base_delay_s` | `float = Field(1.0, ge=0)` | `RETRY_BASE_DELAY_S` |
  | `retry_max_delay_s` | `float = Field(30.0, ge=0)` | `RETRY_MAX_DELAY_S` |
  | `breaker_failure_threshold` | `int = Field(3, ge=1)` | `BREAKER_FAILURE_THRESHOLD` |
  | `breaker_cooldown_s` | `float = Field(30.0, ge=0)` | `BREAKER_COOLDOWN_S` |
  | `cache_ttl_s` | `float = Field(3600.0, ge=0)` | `CACHE_TTL_S` |
  | `chroma_host` | `str \| None = None` | `CHROMA_HOST` |
  | `chroma_port` | `int = 8000` | `CHROMA_PORT` |
  | `log_level` | `str = "INFO"` | `LOG_LEVEL` |

- Produces `Settings.model_for(role: Role) -> str`.
- Produces two properties: `Settings.cache_dir -> Path` (`state_dir / "cache"`) and `Settings.cost_reports_dir -> Path` (`project_dir / "docs" / "cost_reports"`).

- [ ] **Step 1: Write the failing tests** in `tests/test_config.py`.

```python
def test_role_model_defaults(tmp_path):
    s = Settings.from_env(project_dir=tmp_path)
    assert s.model_for("pm") == "anthropic:claude-sonnet-5-5"
    assert s.model_for("coder") == s.model == "anthropic:claude-haiku-5-5"
    assert s.model_for("qa") == "anthropic:claude-haiku-5-5"

def test_role_model_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("PM_MODEL", "anthropic:claude-opus-5-5")
    monkeypatch.setenv("CODER_MODEL", "anthropic:claude-sonnet-5-5")
    monkeypatch.setenv("QA_MODEL", "openai:gpt-4o")
    s = Settings.from_env(project_dir=tmp_path)
    assert (s.model_for("pm"), s.model_for("coder"), s.model_for("qa")) == (
        "anthropic:claude-opus-5-5", "anthropic:claude-sonnet-5-5", "openai:gpt-4o")

def test_retry_max_capped_at_3(tmp_path):
    with pytest.raises(ValidationError):
        Settings.from_env(project_dir=tmp_path, retry_max=4)

def test_resilience_defaults(tmp_path, monkeypatch):
    for v in ("RETRY_BASE_DELAY_S", "RETRY_MAX_DELAY_S", "CACHE_TTL_S"):
        monkeypatch.delenv(v, raising=False)
    s = Settings.from_env(project_dir=tmp_path)
    assert (s.retry_max, s.retry_base_delay_s, s.retry_max_delay_s) == (3, 1.0, 30.0)
    assert (s.breaker_failure_threshold, s.breaker_cooldown_s, s.cache_ttl_s) == (3, 30.0, 3600.0)
    assert s.cache_dir == tmp_path / ".coder" / "cache"
    assert s.cost_reports_dir == tmp_path / "docs" / "cost_reports"
```

  Also update line 10's `assert s.model == "anthropic:claude-sonnet-5-5"` to the Haiku default. This is a deliberate change: name it in the commit.

- [ ] **Step 2: Run the tests.**
  - Run: `uv run pytest tests/test_config.py -v`.
  - Expected: the 4 new tests FAIL.
- [ ] **Step 3: Implement the fields and env mapping.**
  - Add the fields, their `_ENV_VARS` entries, `model_for` and the two properties.
- [ ] **Step 4: Check every role's provider.**
  - `team_cli.main` checks `provider_prerequisite` for each distinct `settings.model_for(r)`, `r` in `("pm", "coder", "qa")`, and prints the first reason.
  - `tests/live/conftest.py` does the same.
  - Add a `--pm-model` and `--qa-model` CLI flag next to `--model`. Help text: "--model: Coder model".
- [ ] **Step 5: Update the env templates.**
  - `.env.example`: list every variable in the table above with defaults or placeholders, plus `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `CHROMA_HOST`, `CHROMA_PORT`.
  - `.env.op`: same keys. `ANTHROPIC_API_KEY` and `LANGSMITH_API_KEY` stay `op://` refs.
- [ ] **Step 6: Run the full suite and commit.**
  - Run: `uv run pytest -q`. Expected: all pass.
  - Commit `feat(config): model per role and resilience settings (CODER_MODEL now Coder-only, default Haiku)`.

### Task 3: Error hierarchy, classification and retry

**Files:**
- Create: `resilience/__init__.py`, `resilience/errors.py`, `resilience/retry.py`
- Test: `tests/test_retry.py`

**Interfaces:**
- Produces in `resilience/errors.py`:
  - `class AgentError(Exception)`: base class with `.original: BaseException | None`. `str()` is `f"{type(original).__name__}: {original}"` when `original` is set.
  - `class TransientError(AgentError)` with `.attempts: int`.
  - `class PermanentError(AgentError)`.
  - `class CircuitOpenError(TransientError)` with `.breaker: str`.
  - `ErrorKind = Literal["transient", "permanent"]`.
  - `classify_llm_error(exc: BaseException) -> ErrorKind`: transient if any of these holds, else permanent:
    - `isinstance(exc, TransientError)`;
    - builtin `TimeoutError` / `ConnectionError`;
    - a `status_code` attribute in `{408, 409, 429}` or `>= 500`, which includes 529;
    - the class name is in `{"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError", "OverloadedError", "ServiceUnavailableError"}`;
    - an `httpx.TimeoutException` / `httpx.TransportError`.
  - `classify_tool_error(exc) -> ErrorKind`: transient for `TransientError`, `TimeoutError`, `ConnectionError`, or an `OSError` whose errno is in `{EAGAIN, EBUSY, EINTR, ETIMEDOUT, EMFILE, ENFILE}`. Everything else is permanent, including `ToolError` and `ValidationError`.
  - `classify_chroma_error(exc) -> ErrorKind`: transient for `TransientError`, connection/timeout errors, `httpx.TransportError`, or a `status_code >= 500` / `== 429`; else permanent.
- Produces in `resilience/retry.py`:
  - `@dataclass(frozen=True) class RetryPolicy`:
    - fields `max_retries: int = 3`, `base_delay_s: float = 1.0`, `max_delay_s: float = 30.0`;
    - `def delay(self, attempt: int, rng: Callable[[float, float], float] = random.uniform) -> float`;
    - `@classmethod from_settings(cls, s: Settings) -> RetryPolicy`.
  - `call_with_retry(fn: Callable[[], T], *, policy: RetryPolicy, classify: Callable[[BaseException], ErrorKind], name: str, sleep: Callable[[float], None] = time.sleep, rng = random.uniform) -> tuple[T, int]`. Returns `(result, attempts)`, where `attempts` counts the original call.
    - Catches only `Exception`.
    - A permanent error raises `PermanentError(original=e)` from `e` with no retry. An `AgentError` that is already permanent is re-raised as is.
    - A transient error retries until `policy.max_retries` retries are used, then raises `TransientError(attempts=…, original=e)`, or re-raises a `CircuitOpenError` unchanged.
    - Logs `retry_attempt` (fields `name`, `attempt`, `error_class`, `delay_s`) and `retry_exhausted` through `observability.logs.log_event`. That module lands in Task 5, so until then call a module-local `_log` shim that Task 5 rewires.

- [ ] **Step 1: Write the failing tests** in `tests/test_retry.py`.

```python
class Flaky(Exception):
    status_code = 429

def counting(fails: int, exc=Flaky):
    calls = {"n": 0}
    def fn():
        calls["n"] += 1
        if calls["n"] <= fails:
            raise exc("busy")
        return "ok"
    return fn, calls

P = RetryPolicy(max_retries=3, base_delay_s=0, max_delay_s=0)

def test_transient_retried_then_succeeds():
    fn, calls = counting(2)
    assert call_with_retry(fn, policy=P, classify=classify_llm_error, name="t", sleep=lambda s: None) == ("ok", 3)

def test_persistent_transient_stops_after_exactly_3_retries():
    fn, calls = counting(99)
    with pytest.raises(TransientError) as ei:
        call_with_retry(fn, policy=P, classify=classify_llm_error, name="t", sleep=lambda s: None)
    assert calls["n"] == 4 and ei.value.attempts == 4 and isinstance(ei.value.original, Flaky)

def test_permanent_not_retried():
    fn, calls = counting(99, exc=ValueError)
    with pytest.raises(PermanentError):
        call_with_retry(fn, policy=P, classify=classify_llm_error, name="t", sleep=lambda s: None)
    assert calls["n"] == 1

def test_full_jitter_bounds():
    p = RetryPolicy(max_retries=3, base_delay_s=1.0, max_delay_s=3.0)
    assert p.delay(0, rng=lambda a, b: b) == 1.0 and p.delay(1, rng=lambda a, b: b) == 2.0
    assert p.delay(5, rng=lambda a, b: b) == 3.0 and p.delay(2, rng=lambda a, b: a) == 0.0

def test_sleeps_use_policy_delays():
    slept = []
    fn, _ = counting(3)
    call_with_retry(fn, policy=RetryPolicy(3, 1.0, 30.0), classify=classify_llm_error, name="t",
                    sleep=slept.append, rng=lambda a, b: b)
    assert slept == [1.0, 2.0, 4.0]

def test_keyboard_interrupt_during_sleep_propagates():  # Review Focus 5
    fn, calls = counting(5)
    def sleep(_):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        call_with_retry(fn, policy=P, classify=classify_llm_error, name="t", sleep=sleep)
    assert calls["n"] == 1

@pytest.mark.parametrize("exc,kind", [
    (TimeoutError(), "transient"), (ConnectionError(), "transient"), (Flaky(), "transient"),
    (type("OverloadedError", (Exception,), {})(), "transient"), (ValueError("bad"), "permanent"),
    (type("E", (Exception,), {"status_code": 401})(), "permanent"),
    (type("E", (Exception,), {"status_code": 529})(), "transient"),
])
def test_classify_llm_error(exc, kind):
    assert classify_llm_error(exc) == kind

def test_classify_tool_error():
    assert classify_tool_error(OSError(errno.EBUSY, "busy")) == "transient"
    assert classify_tool_error(OSError(errno.ENOENT, "gone")) == "permanent"
    assert classify_tool_error(ToolError("bad path")) == "permanent"
```

- [ ] **Step 2: Run the tests.**
  - Run: `uv run pytest tests/test_retry.py -v`.
  - Expected: FAIL with `ModuleNotFoundError: resilience`.
- [ ] **Step 3: Implement** `resilience/errors.py` and `resilience/retry.py` per the Interfaces.
- [ ] **Step 4: Run the tests.**
  - Run: `uv run pytest tests/test_retry.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(resilience): error hierarchy, classifiers and call-level retry with full jitter`.

### Task 4: Circuit breaker, registry and process runtime

**Files:**
- Create: `resilience/breaker.py`, `resilience/runtime.py`
- Modify: `tests/conftest.py` (the autouse fixture calls `resilience.runtime.reset()`)
- Test: `tests/test_breaker.py`

**Interfaces:**
- Produces in `resilience/breaker.py`:
  - `class BreakerState(StrEnum)` with `CLOSED = "closed"`, `OPEN = "open"`, `HALF_OPEN = "half_open"`.
  - `class CircuitBreaker`:
    - `__init__(self, name: str, *, failure_threshold: int = 3, cooldown_s: float = 30.0, clock: Callable[[], float] = time.monotonic)`;
    - `state: BreakerState`, a property that reports `HALF_OPEN` once the cool-down has elapsed;
    - `failures: int`;
    - `def call(self, fn: Callable[[], T]) -> T`.
  - Semantics of `call`:
    - **OPEN** before the cool-down: raise `CircuitOpenError(breaker=name)` without calling `fn`.
    - After the cool-down: move to HALF_OPEN and allow exactly one in-flight probe, guarded by a `threading.Lock`. Concurrent callers get `CircuitOpenError`.
    - On `TransientError` (only after `fn`'s retries are exhausted): `failures += 1`. A failed probe moves HALF_OPEN → OPEN. In CLOSED, reaching the threshold moves to OPEN. Then re-raise.
    - On success: `failures = 0`, and HALF_OPEN → CLOSED.
    - Any other exception, including `PermanentError` and `KeyboardInterrupt`: release the probe slot, leave the count and state unchanged, re-raise.
    - Every transition calls `log_event("breaker_transition", level=WARNING, breaker=name, from_state=…, to_state=…, failures=…)`.
  - `class BreakerRegistry`:
    - `get(self, name: str) -> CircuitBreaker` creates the breaker lazily from the configured threshold, cool-down and clock;
    - `configure(self, *, failure_threshold: int, cooldown_s: float, clock=time.monotonic) -> None` applies to breakers created later and updates existing ones;
    - `reset(self) -> None` drops them all.
- Produces in `resilience/runtime.py`:
  - `@dataclass class Resilience`: `policy: RetryPolicy`, `breakers: BreakerRegistry`, `sleep: Callable[[float], None] = time.sleep`.
  - `configure(settings: Settings) -> None`: sets the policy from settings and configures the registry.
  - `runtime() -> Resilience`.
  - `reset() -> None`: defaults with zero delays, new registry.
  - `resilient_call(fn: Callable[[], T], *, breaker: str, classify, name: str) -> tuple[T, int]`, i.e. `runtime().breakers.get(breaker).call(lambda: call_with_retry(fn, policy=…, classify=classify, name=name, sleep=runtime().sleep))`.

- [ ] **Step 1: Write the failing tests** in `tests/test_breaker.py`. Use a fake clock `t = [0.0]; clock = lambda: t[0]` and helpers `fail = lambda: (_ for _ in ()).throw(TransientError("x"))` and `ok = lambda: "ok"`.

```python
def test_opens_after_3_consecutive_transient_failures_and_blocks(caplog):
    b = CircuitBreaker("pm", failure_threshold=3, cooldown_s=30, clock=clock)
    for _ in range(3):
        with pytest.raises(TransientError):
            b.call(fail)
    assert b.state is BreakerState.OPEN
    called = []
    with pytest.raises(CircuitOpenError):
        b.call(lambda: called.append(1))
    assert called == []
    assert [r for r in caplog.records if r.levelname == "WARNING" and "breaker_transition" in r.getMessage()]

def test_half_open_after_cooldown_probe_success_closes(): ...   # t[0] = 30.0 → state HALF_OPEN; b.call(ok) == "ok"; state CLOSED; failures 0
def test_half_open_probe_failure_reopens(): ...                 # probe raises TransientError → OPEN; next call before +30 s → CircuitOpenError
def test_half_open_allows_single_probe(): ...                   # inside a probe fn, a nested b.call(ok) raises CircuitOpenError
def test_success_resets_count(): ...                            # fail, fail, ok, fail, fail → still CLOSED
def test_permanent_errors_neither_count_nor_reset(): ...        # fail, fail, PermanentError ×5, fail → OPEN (count 3); PermanentError never opens alone
def test_parsing_errors_do_not_count(): ...                     # ValueError ×10 → CLOSED
def test_cooldown_configurable(): ...                           # cooldown_s=5: t=4.9 → OPEN, t=5.0 → HALF_OPEN
def test_registry_shares_breaker_per_name(): ...                # registry.get("pm") is registry.get("pm"); "qa" differs
```

  Each `...` test needs real assertions as its comment states. Write them out in full.

- [ ] **Step 2: Run the tests.**
  - Run: `uv run pytest tests/test_breaker.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `resilience/breaker.py` and `resilience/runtime.py`. Extend the conftest fixture to call `resilience.runtime.reset()`.
- [ ] **Step 4: Run the tests.**
  - Run: `uv run pytest tests/test_breaker.py tests/test_retry.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(resilience): per-name circuit breakers (transient-only counting, single half-open probe)`.

### Task 5: JSON logs, run context, redaction

**Files:**
- Create: `observability/__init__.py`, `observability/logs.py`, `observability/context.py`
- Modify: `resilience/retry.py`, `resilience/breaker.py` (point `_log` at `log_event`)
- Test: `tests/test_logs.py`

**Interfaces:**
- Produces in `observability/logs.py`:
  - `LOGGER_NAME = "mad"`.
  - `class JsonFormatter(logging.Formatter)`: one JSON object per record, with keys `ts` (ISO-8601 UTC), `level`, `run_id`, `agent`, `event`, plus `record.fields`.
  - `configure_logging(level: str = "INFO") -> None`: an idempotent stderr `StreamHandler` with `JsonFormatter`, and `propagate = True` so `caplog` sees records.
  - `log_event(event: str, *, level: int = logging.INFO, **fields) -> None`: fills `run_id` / `agent` from `observability.context` and passes every string field through `redact`.
  - `redact(text: str) -> str`: replaces the value of every env var whose name ends with `_API_KEY`, `_TOKEN` or `_SECRET` (values of 8+ characters) with `"[redacted]"`.
  - `run_log_file(path: Path) -> ContextManager[None]`: attaches a `FileHandler` with `JsonFormatter` for the duration and removes it after.
- Produces in `observability/context.py`:
  - `@dataclass class RunContext`: `run_id: str`, `calls: list[CallRecord]`, `tools: list[ToolRecord]`, `trace_url: str | None = None`, `lock: threading.Lock`.
  - `run_scope(ctx: RunContext) -> ContextManager[RunContext]`.
  - `current_run() -> RunContext | None`.
  - `agent_scope(name: str) -> ContextManager[None]`.
  - `current_agent() -> str | None`.
  - `record_call(rec: CallRecord) -> None` and `record_tool(rec: ToolRecord) -> None`: append under the lock, and are no-ops with no run.
  - Two `ContextVar`s. `CallRecord` and `ToolRecord` come from Task 6. Use `TYPE_CHECKING` imports or define Task 6's models first; either order works.

- [ ] **Step 1: Write the failing tests.**

```python
def test_event_is_one_json_line_with_context(tmp_path):
    ctx = RunContext(run_id="abc123")
    with run_scope(ctx), agent_scope("qa"), run_log_file(tmp_path / "events.jsonl"):
        log_event("cache_hit", key="k1")
    rec = json.loads((tmp_path / "events.jsonl").read_text().strip())
    assert rec["event"] == "cache_hit" and rec["run_id"] == "abc123" and rec["agent"] == "qa"
    assert rec["level"] == "INFO" and rec["key"] == "k1" and "ts" in rec

def test_secret_values_redacted(tmp_path, monkeypatch):  # Review Focus 4
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecret-123")
    with run_log_file(tmp_path / "e.jsonl"):
        log_event("llm_error", message="401 invalid key sk-ant-supersecret-123")
    assert "supersecret" not in (tmp_path / "e.jsonl").read_text()
    assert redact("x sk-ant-supersecret-123 y") == "x [redacted] y"

def test_file_handler_removed_after_scope(tmp_path):
    with run_log_file(tmp_path / "a.jsonl"):
        pass
    log_event("after")
    assert (tmp_path / "a.jsonl").read_text() == ""

def test_retry_logs_attempts(caplog): ...  # run Task 3's counting(2) through call_with_retry; two "retry_attempt" records with attempt 1, 2
```

- [ ] **Step 2: Run the tests.**
  - Run: `uv run pytest tests/test_logs.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement both modules** and rewire `_log` in retry and breaker.
- [ ] **Step 4: Run the tests.**
  - Run: `uv run pytest tests/test_logs.py tests/test_retry.py tests/test_breaker.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(observability): JSON event logs with run/agent context and secret redaction`.

### Task 6: Pricing, call records, cost report

**Files:**
- Create: `observability/costs.py`
- Test: `tests/test_costs.py`

**Interfaces:**
- Produces:
  - `class Price(BaseModel)`: `input`, `output`, `cache_read`, `cache_write`, all `float` USD per 1M tokens.
  - `PRICING: dict[str, Price]`, keyed by the model name without the provider prefix (`"claude-sonnet-5-5"`, `"claude-haiku-5-5"`, plus `"claude-opus-5-5"`).
  - `PRICING_AS_OF: str`.
  - `class CallRecord(BaseModel)`: `ts: str`, `role: str`, `agent: str | None`, `model: str`, `prompt_tokens: int`, `completion_tokens: int`, `cache_read_tokens: int`, `cache_write_tokens: int`, `cache_hit: bool`, `cost_usd: float`, `priced: bool`, `duration_ms: float`, `attempts: int`.
  - `class ToolRecord(BaseModel)`: `ts: str`, `agent: str | None`, `tool: str`, `ok: bool`, `duration_ms: float`, `attempts: int`.
  - `usage_from_message(msg: AIMessage) -> tuple[int, int, int, int]`: `(prompt, completion, cache_read, cache_write)` from `usage_metadata`, with `input_token_details.cache_read` / `cache_creation`, and zeros if absent.
  - `cost_usd(model: str, prompt: int, completion: int, cache_read: int, cache_write: int) -> tuple[float, bool]`. LangChain's `input_tokens` already includes the cache tokens, so it charges `(prompt - cache_read - cache_write) * input + cache_read * cache_read_price + cache_write * cache_write_price + completion * output`, all per 1e6. An unknown model returns `(0.0, False)` and logs `unpriced_model` at WARNING.
  - `build_cost_report(*, run_id: str, started_at: str, finished_at: str, requirement: str, outcome: Literal["success", "unresolved", "failed", "aborted"], models: dict[str, str], calls: list[CallRecord], trace_url: str | None, error: dict | None = None) -> dict`. Keys (§5.5):
    - `run_id`, `started_at`, `finished_at`, `requirement`, `outcome`, `error`;
    - `models`, `pricing` (the `Price` entries used, plus `as_of`), `trace_url`;
    - `calls`;
    - `agents.{pm,coder,qa}` and `total`, each with `{calls, prompt_tokens, completion_tokens, cache_read_tokens, cache_write_tokens, cache_hits, cost_usd}`;
    - `unpriced_models`.
  - `write_cost_report(report: dict, reports_dir: Path, run_dir: Path) -> Path`: writes `reports_dir/<run_id>.json` (creating the dir) and `run_dir/cost-report.json`, and returns the first path.

- [ ] **Step 1: Fill the prices from current Anthropic pricing.**
  - Invoke the `claude-api` skill, not memory. Use input, output, cache read, and 5-minute cache write per MTok for the three models.
  - Set `PRICING_AS_OF = "2026-11-08"` and add a comment naming the source.
- [ ] **Step 2: Write the failing tests.**

```python
def rec(role, p, c, cost, hit=False):
    return CallRecord(ts="t", role=role, agent=role, model="anthropic:claude-haiku-5-5", prompt_tokens=p,
                      completion_tokens=c, cache_read_tokens=0, cache_write_tokens=0, cache_hit=hit,
                      cost_usd=cost, priced=True, duration_ms=1.0, attempts=1)

def test_cost_from_price_table():
    pr = PRICING["claude-haiku-5-5"]
    cost, priced = cost_usd("anthropic:claude-haiku-5-5", 1_000_000, 1_000_000, 0, 0)
    assert priced and cost == pytest.approx(pr.input + pr.output)

def test_cache_tokens_priced_separately():
    pr = PRICING["claude-sonnet-5-5"]
    cost, _ = cost_usd("anthropic:claude-sonnet-5-5", 1_000_000, 0, 400_000, 100_000)
    assert cost == pytest.approx(0.5 * pr.input + 0.4 * pr.cache_read + 0.1 * pr.cache_write)

def test_unknown_model_costs_zero_and_flags(caplog):
    assert cost_usd("ollama:llama3", 10, 10, 0, 0) == (0.0, False)

def test_usage_from_message():
    m = AIMessage("x", usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150,
                                       "input_token_details": {"cache_read": 100, "cache_creation": 0}})
    assert usage_from_message(m) == (120, 30, 100, 0)

def test_report_totals_reconcile_with_calls():
    calls = [rec("pm", 100, 50, 0.01), rec("coder", 200, 80, 0.02), rec("qa", 10, 5, 0.0, hit=True)]
    r = build_cost_report(run_id="r1", started_at="a", finished_at="b", requirement="q", outcome="success",
                          models={"pm": "m"}, calls=calls, trace_url=None)
    assert r["total"]["prompt_tokens"] == sum(c.prompt_tokens for c in calls) == 310
    assert r["total"]["cost_usd"] == pytest.approx(sum(a["cost_usd"] for a in r["agents"].values()))
    assert r["agents"]["qa"]["cache_hits"] == 1 and len(r["calls"]) == 3

def test_write_cost_report_two_copies(tmp_path): ...  # both files exist, identical JSON, returned path is reports_dir/r1.json
```

- [ ] **Step 3: Run the tests.**
  - Run: `uv run pytest tests/test_costs.py -v`. Expected: FAIL.
- [ ] **Step 4: Implement** `observability/costs.py`.
- [ ] **Step 5: Run the tests.**
  - Run: `uv run pytest tests/test_costs.py -v`. Expected: PASS.
- [ ] **Step 6: Commit** `feat(observability): per-call cost records, pricing table and reconciling cost report`.

### Task 7: Response cache with TTL

**Files:**
- Create: `resilience/cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- `class ResponseCache`:
  - `__init__(self, root: Path, ttl_s: float, clock: Callable[[], float] = time.time)`;
  - `enabled: bool`, which is `ttl_s > 0`;
  - `key(self, model_id: str, messages: list[BaseMessage], bound: dict, stop: list[str] | None) -> str`: SHA-256 hex of a canonical JSON with model id, `messages_to_dict(messages)`, `bound` (default `str`) and stop, using `sort_keys=True`;
  - `get(self, key: str) -> tuple[Literal["hit", "miss", "expired"], AIMessage | None]`;
  - `put(self, key: str, message: AIMessage) -> None`.
- Storage: one file per key, `root/<key[:2]>/<key>.json`, holding `{"created_at": float, "message": messages_to_dict([m])[0]}`. Writes go to a temp file in the same dir, then `os.replace`. An unreadable or invalid file is a `"miss"`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_hit_within_ttl(tmp_path):
    t = [1000.0]
    c = ResponseCache(tmp_path, ttl_s=60, clock=lambda: t[0])
    k = c.key("m", [HumanMessage("hi")], {}, None)
    assert c.get(k) == ("miss", None)
    c.put(k, AIMessage("cached"))
    t[0] += 59
    status, msg = c.get(k)
    assert status == "hit" and msg.content == "cached"

def test_expired_after_ttl(tmp_path): ...       # t += 60.01 → ("expired", None)
def test_key_changes_with_messages_tools_model(tmp_path): ...  # 3 different keys; identical inputs → same key
def test_disabled_when_ttl_zero(tmp_path): ...  # ResponseCache(tmp_path, 0).enabled is False

def test_corrupt_or_partial_file_is_a_miss(tmp_path):  # Review Focus 2
    c = ResponseCache(tmp_path, ttl_s=60)
    k = c.key("m", [HumanMessage("hi")], {}, None)
    c.put(k, AIMessage("ok"))
    path = next(tmp_path.rglob(f"{k}.json"))
    path.write_text('{"created_at": 1, "mess')
    assert c.get(k) == ("miss", None)
    c.put(k, AIMessage("again"))
    assert c.get(k)[0] == "hit"
```

- [ ] **Step 2: Run the tests.**
  - Run: `uv run pytest tests/test_cache.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `resilience/cache.py`.
- [ ] **Step 4: Run the tests.**
  - Run: `uv run pytest tests/test_cache.py -v`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(resilience): on-disk response cache with TTL and atomic writes`.

### Task 8: ResilientChatModel at the LLM call boundary

**Files:**
- Modify: `agents/model.py`, `agents/pm_agent.py`, `agents/coder_agent.py`, `agents/qa_agent.py`
- Test: `tests/test_resilient_model.py`

**Interfaces:**
- Consumes:
  - Task 3: `call_with_retry`, `classify_llm_error`;
  - Task 4: `runtime()`, `configure`;
  - Task 5: `log_event`, `current_agent`;
  - Task 6: `CallRecord`, `cost_usd`, `usage_from_message`, `record_call`;
  - Task 7: `ResponseCache`.
- Produces in `agents/model.py`:
  - `class ResilientChatModel(BaseChatModel)`. Pydantic fields: `inner: BaseChatModel`, `role: Role`, `model_id: str`, `cache: ResponseCache | None = None`, `bound: dict = {}`, `cacheable: bool = False`, with `arbitrary_types_allowed`.
    - `bind_tools(self, tools, **kwargs) -> ResilientChatModel`:
      - calls `self.inner.bind_tools(tools, **kwargs)`;
      - if the result is a `RunnableBinding`, merges its `.kwargs` into `bound`, so the provider formats tools and `tool_choice` itself;
      - otherwise (e.g. `ScriptedChatModel` returns itself) leaves `bound` unchanged;
      - returns a `model_copy`.
    - `cached(self) -> ResilientChatModel`: a copy with `cacheable=True`.
    - `_generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult`:
      1. Forwarded kwargs are `{**bound, **kwargs}` minus any key starting with `ls_`.
      2. For an `anthropic:` model, the first `SystemMessage` and the last bound tool get `cache_control: {"type": "ephemeral"}`.
      3. `breaker = runtime().breakers.get(self.role)`.
      4. If the breaker is OPEN and the call is cacheable with an unexpired entry, return it and log `degraded_fallback_cached`. If OPEN with no entry, let the breaker raise `CircuitOpenError`.
      5. If cacheable and the cache is enabled: `cache.get(key)`, logging `cache_hit` / `cache_miss` / `cache_expired`. A hit records a `CallRecord` with zero tokens, `cache_hit=True` and `cost_usd=0`, and returns.
      6. `breaker.call(lambda: call_with_retry(lambda: self.inner._generate(messages, stop=stop, run_manager=None, **fwd), …))`.
      7. Record a `CallRecord`. `agent` is `current_agent() or role`. Log `llm_call` with model, tokens, cost, cache_hit, duration_ms, attempts.
      8. On a cacheable success, `cache.put`.
    - `_llm_type`: `"resilient"`.
  - `make_model(settings: Settings, role: Role = "coder") -> ResilientChatModel`:
    - `init_chat_model(settings.model_for(role), max_tokens=settings.max_tokens, **({"max_retries": 0} if provider in {"anthropic", "openai"} else {}))`;
    - wrapped with `resilient(...)`.
  - `resilient(model: BaseChatModel, settings: Settings, role: Role, model_id: str | None = None) -> ResilientChatModel`:
    - returns `model` unchanged if it is already resilient;
    - calls `resilience.runtime.configure(settings)`;
    - builds the cache from `ResponseCache(settings.cache_dir, settings.cache_ttl_s)`;
    - `model_id` defaults to `settings.model_for(role)`.
- Changes to the agents:
  - `PMAgent`, `CoderAgent` and `QAAgent` wrap an injected `model` with `resilient(model, self.settings, "<role>")` in `__init__`, and call `make_model(self.settings, "<role>")` otherwise.
  - `PMAgent._structured` and `QAAgent._structured` use `self.model.cached().with_structured_output(schema, include_raw=True)`.
  - The Coder's `_bound_model` and `critique` stay uncached.

- [ ] **Step 1: Write the failing tests.** Use `ScriptedChatModel` from `tests/fakes.py` as `inner`. A `Flaky` exception has `status_code = 429`.

```python
def wrap(settings, *responses, role="pm"):
    return resilient(ScriptedChatModel(responses=list(responses)), settings, role)

def test_transient_llm_error_retried_once_then_succeeds(settings):
    m = wrap(settings, Flaky("429"), ai("hello"))
    assert m.invoke([HumanMessage("x")]).content == "hello"

def test_retry_does_not_replay_agent_loop(settings):
    inner = ScriptedChatModel(responses=[Flaky("1"), *happy()])  # tests/fakes.py:happy (2 ReAct turns)
    a = CoderAgent(settings, model=inner, embedding_function=FakeEmbedding())
    assert a.run("t").status == "success"
    assert len(inner.calls) == 3  # 1 failed attempt + 2 turns: the retry repeated one call, not the loop

def test_persistent_failure_raises_after_3_retries_and_counts_toward_breaker(settings):
    inner = ScriptedChatModel(responses=[Flaky("x")] * 4)
    with pytest.raises(TransientError):
        resilient(inner, settings, "pm").invoke([HumanMessage("x")])
    assert len(inner.calls) == 4 and runtime().breakers.get("pm").failures == 1

def test_structured_output_with_include_raw_through_wrapper(settings):
    m = wrap(settings, spec_response(make_spec()))
    out = m.cached().with_structured_output(Spec, include_raw=True).invoke([HumanMessage("req")])
    assert out["parsing_error"] is None and isinstance(out["parsed"], Spec)

def test_bind_tools_delegates_formatting_to_inner(settings, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from langchain_anthropic import ChatAnthropic
    m = resilient(ChatAnthropic(model="claude-haiku-5-5", max_retries=0), settings, "coder")
    b = m.bind_tools(TOOL_SPECS)
    assert isinstance(b, ResilientChatModel) and b.bound["tools"][0]["name"] == TOOL_SPECS[0]["name"]

def test_cache_hit_within_ttl_records_zero_cost(settings_with_cache):  # settings with cache_ttl_s=60
    ctx = RunContext(run_id="r")
    with run_scope(ctx):
        m = wrap(settings_with_cache, spec_response(make_spec()))
        for _ in range(2):
            m.cached().with_structured_output(Spec, include_raw=True).invoke([HumanMessage("same")])
    assert [c.cache_hit for c in ctx.calls] == [False, True] and ctx.calls[1].cost_usd == 0

def test_uncached_calls_never_hit_cache(settings_with_cache): ...  # same as above without .cached(): 2 scripted responses consumed, no cache_hit
def test_open_circuit_returns_cached_or_structured_error(settings_with_cache): ...
    # 1. warm the cache with a cacheable call
    # 2. force runtime().breakers.get("pm") OPEN (3 TransientErrors via breaker.call)
    # 3. the cached call returns the cached message and logs degraded_fallback_cached
    # 4. a different prompt raises CircuitOpenError, and inner is never called
def test_usage_recorded_after_every_call(settings): ...  # inner returns AIMessage with usage_metadata; ctx.calls[0].prompt_tokens == 120 etc.
def test_make_model_uses_role_model_and_disables_sdk_retries(settings, monkeypatch):
    ...  # monkeypatch agents.model.init_chat_model to capture args: model_for("pm") and max_retries == 0
```

  Add the `settings_with_cache` fixture to the test module: `Settings.from_env(project_dir=tmp_path, cache_ttl_s=60)`.

- [ ] **Step 2: Run the tests.**
  - Run: `uv run pytest tests/test_resilient_model.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `ResilientChatModel`, `make_model` and `resilient`, then change the three agents.
- [ ] **Step 4: Run the new tests and the full suite.**
  - Run: `uv run pytest tests/test_resilient_model.py -v && uv run pytest -q`.
  - Expected: all pass. v3 tests that script `RuntimeError(...)` still behave the same: it's classified permanent, so there's no retry, and the `str()` keeps the original text.
- [ ] **Step 5: Commit** `feat(model): ResilientChatModel (retry, role breaker, response cache, cost records, prompt caching) at every LLM call`.

### Task 9: Tool and Chroma resilience, idempotent memory writes, Chroma server client

**Files:**
- Create: `tools/guarded.py`
- Modify: `tools/dispatch.py`, `agents/qa_agent.py`, `memory/semantic.py`, `agents/coder_agent.py`, `app/team_cli.py`
- Test: `tests/test_dispatch.py`, `tests/test_memory.py`, `tests/test_qa_agent.py`, `tests/test_team_cli.py`

**Interfaces:**
- Produces `guarded_tool(name: str, fn: Callable[[], T]) -> T` in `tools/guarded.py`:
  - `resilient_call(fn, breaker="tools", classify=classify_tool_error, name=name)`;
  - records a `ToolRecord` (agent, tool, ok, duration_ms, attempts) and logs `tool_call`;
  - re-raises.
- `run_action`:
  - wraps `_dispatch` in `guarded_tool(name, …)`;
  - converts an escaping `TransientError` (including `CircuitOpenError`) into `ToolError(f"{name} unavailable: {e}")`, so the Coder sees an observation;
  - existing `ToolError` / `OSError` handling is unchanged.
- QA direct calls:
  - `_pytest` → `guarded_tool("exec_python", lambda: exec_python(...))`;
  - `_restore` / `_save_suite` → `guarded_tool("write_file", …)`;
  - before calling, validate the arguments with `ARG_MODELS["write_file"].model_validate({"path": …, "content": …})`, and for exec, check `mode` is `"script"` and `0 < timeout <= settings.exec_timeout_s`;
  - a `TransientError` here surfaces as the existing QA crash → `qa_error` path.
- `SemanticMemory`:
  - `__init__(self, chroma_dir: Path, embedding_function: Any = None, collection: str = "coder_runs", *, host: str | None = None, port: int = 8000)`: uses `chromadb.HttpClient(host=host, port=port)` when `host` is set, else `PersistentClient`. Client creation goes through `resilient_call(..., breaker="chroma", classify=classify_chroma_error)`. A final failure raises `MemoryUnavailableError(TransientError)`, whose message names `host:port`.
  - `memory_id(text: str, metadata: dict) -> str`: `sha256(text + "\0" + json.dumps(metadata, sort_keys=True)).hexdigest()[:32]`.
  - `add(self, text: str, metadata: dict, *, doc_id: str | None = None) -> str`: fixes `doc_id = doc_id or memory_id(text, metadata)` **before** the retry, then `resilient_call(lambda: self._collection.upsert(ids=[doc_id], …), breaker="chroma", …)`.
  - `search`, `count`, `clear`, `reopen`: each goes through `resilient_call(..., breaker="chroma")`.
- `CoderAgent` passes `host=settings.chroma_host, port=settings.chroma_port`.
- `team_cli.main` catches `MemoryUnavailableError` around `TeamRunner(settings)` and prints `team: cannot run: {e}`. Return 2.

- [ ] **Step 1: Write the failing tests.**

```python
# tests/test_memory.py
def test_add_is_idempotent_when_response_lost_after_commit(tmp_path):
    mem = SemanticMemory(tmp_path / "c", embedding_function=FakeEmbedding())
    real = mem._collection
    state = {"n": 0}
    class LossyCollection:
        def __getattr__(self, a): return getattr(real, a)
        def upsert(self, **kw):
            real.upsert(**kw)
            state["n"] += 1
            if state["n"] == 1:
                raise ConnectionError("response lost")
    mem._collection = LossyCollection()
    doc_id = mem.add("Task: x", {"run_id": "r1"})
    assert state["n"] == 2 and mem.count() == 1 and doc_id == memory_id("Task: x", {"run_id": "r1"})

def test_chroma_unreachable_raises_memory_unavailable(tmp_path):
    with pytest.raises(MemoryUnavailableError, match="127.0.0.1:9"):
        SemanticMemory(tmp_path / "c", host="127.0.0.1", port=9)

# tests/test_team_cli.py  (Review Focus 3)
def test_team_exits_2_when_chroma_unreachable(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHROMA_HOST", "127.0.0.1"); monkeypatch.setenv("CHROMA_PORT", "9")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test"); monkeypatch.setenv("CODER_PROJECT_DIR", str(tmp_path))
    assert team_cli.main(["do x"]) == 2
    err = capsys.readouterr().err
    assert "cannot run" in err and "Chroma" in err and "Traceback" not in err

# tests/test_dispatch.py
def test_transient_os_error_retried_then_succeeds(ctx, monkeypatch): ...  # patch tools.dispatch.write_file to raise OSError(EBUSY) once; run_action succeeds; ToolRecord attempts == 2
def test_persistent_tool_failure_becomes_tool_error_after_bound(ctx, monkeypatch): ...  # EBUSY ×4 → ToolError "unavailable"; called 4 times
def test_program_failure_is_a_result_not_retried(ctx): ...  # exec code "raise SystemExit(3)" → ok observation with exit code 3, exec_python called once

# tests/test_qa_agent.py
def test_qa_direct_tool_args_validated(settings): ...  # a suite whose test_file path escapes the workspace never reaches write_file; QA reports qa_error
```

  For the unreachable-Chroma tests, the zero-delay retry from conftest keeps them fast.

- [ ] **Step 2: Run the tests.**
  - Run: `uv run pytest tests/test_memory.py tests/test_dispatch.py tests/test_qa_agent.py tests/test_team_cli.py -v`.
  - Expected: the new tests FAIL.
- [ ] **Step 3: Implement** per the Interfaces.
- [ ] **Step 4: Run the full suite.**
  - Run: `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `feat(tools,memory): retry+breaker on every tool and Chroma call, idempotent upsert writes, Chroma HttpClient`.

### Task 10: Secret-free sandbox environment

**Files:**
- Modify: `tools/exec_python.py`
- Test: `tests/test_security.py`

**Interfaces:**
- Produces `child_env(record_dir: Path) -> dict[str, str]` in `tools/exec_python.py`. It is the only env given to `subprocess.Popen` (Task 14 reuses it).
- What it keeps, from `os.environ` when present:
  - `PATH`, `HOME`, `LANG`, `LC_ALL` and every `LC_*`;
  - `TMPDIR`, `TEMP`, `TMP`;
  - `SYSTEMROOT`, `COMSPEC` (Windows);
  - `PYTHONPATH`, `VIRTUAL_ENV`.
- Plus `PYTHONPYCACHEPREFIX = str(record_dir / "pycache")`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_generated_code_cannot_see_api_keys(settings, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-should-not-leak")
    ws = Workspace(settings.workspace_dir)
    out = exec_python(ws, record_dir=settings.state_dir / "exec", mode="script",
                      code="import os, json\nprint(json.dumps(sorted(os.environ)))")
    keys = json.loads(out.stdout)
    assert "ANTHROPIC_API_KEY" not in keys and "LANGSMITH_API_KEY" not in keys and "PATH" in keys

def test_child_env_is_allowlist_only(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_TOKEN", "x" * 20)
    assert "SOME_TOKEN" not in child_env(tmp_path)
```

- [ ] **Step 2: Run the tests.**
  - Run: `uv run pytest tests/test_security.py -v`. Expected: the first test FAILS (keys visible).
- [ ] **Step 3: Implement** `child_env` and use it in `exec_python`.
- [ ] **Step 4: Run the tests.**
  - Run: `uv run pytest tests/test_security.py tests/test_exec_python.py tests/test_qa_agent.py -q`.
  - Expected: PASS. QA's pytest subprocess still imports pytest via `sys.executable`'s venv.
- [ ] **Step 5: Commit** `fix(security): generated code runs with an allowlisted environment (no API keys)`.

### Task 11: TeamRunner: run context, events.jsonl, cost report on every outcome

**Files:**
- Modify: `orchestration/team_runner.py`, `app/team_cli.py`
- Test: `tests/test_team_runner.py`, `tests/test_team_cli.py`

**Interfaces:**
- Consumes:
  - Task 5: `RunContext`, `run_scope`, `run_log_file`, `log_event`, `configure_logging`;
  - Task 6: `build_cost_report`, `write_cost_report`;
  - Task 12's `root_trace`. Until Task 12 lands, use a `contextlib.nullcontext(None)` stand-in.
- Produces:
  - `OUTCOME = {"completed": "success", "partial": "unresolved", "failed": "failed"}` in `orchestration/team_runner.py`.
  - `TeamRunner.cost_report_path: Path | None`.
  - Each run dir gains `events.jsonl` and `cost-report.json`.

**`stream()` algorithm.** These are new decisions. Keep the existing body inside the `try`.

```
with RunLock(s.state_dir):                      # busy → WorkspaceBusyError, no run dir, no report (unchanged)
    run_id, run_dir … mkdir                     # as today
    ctx = RunContext(run_id=run_id); started = utc_now_iso(); values = {}; outcome = None; error = None
    with run_scope(ctx), run_log_file(run_dir / "events.jsonl"):
        try:
            log_event("run_start", requirement_chars=len(requirement), models={r: s.model_for(r) for r in ROLES})
            with root_trace(run_id, requirement) as trace_url:
                ctx.trace_url = trace_url; if trace_url: log_event("trace", url=trace_url)
                … existing body (start event, graph.stream loop updating `values`, checks) …
                state = state_to_jsonable(values); outcome = OUTCOME[state["status"]]
                write final-state.json; self.final_state = state
                yield final event                # outcome already set: a consumer closing after `final` is not an abort
        except BaseException as e:               # incl. GeneratorExit and KeyboardInterrupt
            if outcome is None:
                outcome = "aborted"; error = {"type": type(e).__name__, "message": redact(str(e))}
                log_event("run_aborted", level=ERROR, **error)
                write final-state.json from state_to_jsonable(values or initial_project_state(requirement))
            raise
        finally:
            report = build_cost_report(... calls=ctx.calls, outcome=outcome, error=error, trace_url=ctx.trace_url ...)
            self.cost_report_path = write_cost_report(report, s.cost_reports_dir, run_dir)
            log_event("run_end", outcome=outcome, cost_usd=report["total"]["cost_usd"], report=str(self.cost_report_path))
```

- `team_cli.main` calls `configure_logging(settings.log_level)` once, and prints `cost report: <path>` after the summary.
- The existing `KeyboardInterrupt` branch still returns 130.

- [ ] **Step 1: Write the failing tests** in `tests/test_team_runner.py`.

```python
REQ = "Implement a linked list"  # any text; the scripts ignore it

def report_of(runner) -> dict:  # also imported by tests/test_tracing.py
    return json.loads(runner.cost_report_path.read_text())

def test_cost_report_written_on_success_and_reconciles(settings):
    runner = scripted_runner(settings, passing_scripts())
    runner.run(REQ)
    r = report_of(runner)
    assert runner.cost_report_path == settings.cost_reports_dir / f"{r['run_id']}.json"
    assert (runner.run_dir / "cost-report.json").read_text() == runner.cost_report_path.read_text()
    assert r["outcome"] == "success" and r["total"]["calls"] == len(r["calls"]) > 0
    assert {c["role"] for c in r["calls"]} == {"pm", "coder", "qa"}

def test_cost_report_on_unresolved(settings): ...    # nonconverging_scripts(2), qa_max_iterations=2 → outcome "unresolved"
def test_cost_report_on_failed_pm(settings): ...     # bad_spec_responses() → outcome "failed"

@pytest.mark.parametrize("where", ["node", "stream", "post_check"])
def test_cost_report_on_abort(settings, where, monkeypatch):
    runner = scripted_runner(settings, passing_scripts())
    if where == "node":
        monkeypatch.setattr(runner.qa, "review", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))  # Review Focus 1
    elif where == "stream":
        monkeypatch.setattr(TeamEvent, "model_validate", classmethod(lambda cls, d: (_ for _ in ()).throw(RuntimeError("bad event"))))
    kw = {"demo_check": lambda ws, st: (_ for _ in ()).throw(RuntimeError("check crashed"))} if where == "post_check" else {}
    with pytest.raises((KeyboardInterrupt, RuntimeError)):
        for _ in runner.stream(REQ, **kw):
            pass
    r = report_of(runner)
    assert r["outcome"] == "aborted" and r["error"]["type"] in ("KeyboardInterrupt", "RuntimeError")
    assert r["total"]["calls"] == len(r["calls"])
    assert (runner.run_dir / "final-state.json").is_file()

def test_generator_closed_early_is_aborted(settings): ...  # take the first 3 events, gen.close() → outcome "aborted"
def test_events_jsonl_has_run_start_llm_calls_run_end(settings): ...  # events: run_start first, run_end last, ≥1 llm_call, all with the same run_id
def test_busy_lock_writes_no_report(settings): ...  # existing test_busy_lock_no_run_dir also asserts no file in settings.cost_reports_dir
```

  The `qa.review` node error case: `qa_node` catches `Exception` but not `KeyboardInterrupt`, so the latter escapes. Confirm the `where="node"` case raises.

  In `tests/test_team_cli.py`, add `test_team_ctrl_c_exits_130_with_report` (Review Focus 1). Patch the runner so a node raises `KeyboardInterrupt`. Expect exit code 130, an aborted report file, and no `Traceback` in stderr.

- [ ] **Step 2: Run the tests.**
  - Run: `uv run pytest tests/test_team_runner.py tests/test_team_cli.py -v`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement** the algorithm above.
- [ ] **Step 4: Run the full suite.**
  - Run: `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `feat(runner): per-run context, events.jsonl, cost report on success/unresolved/failed/aborted`.

### Task 12: LangSmith tracing: one root, agent spans with the four fields

**Files:**
- Create: `observability/tracing.py`
- Modify: `orchestration/team_graph.py` (the `add_node` calls at lines 361–367), `orchestration/team_runner.py` (replace the stand-in)
- Test: `tests/test_tracing.py`

**Interfaces:**
- `tracing_enabled() -> bool`: `langsmith.utils.tracing_is_enabled()` and `LANGSMITH_API_KEY` is set.
- `root_trace(run_id: str, requirement: str) -> ContextManager[str | None]`:
  - when tracing is on: `langsmith.trace(name=f"team-run:{run_id}", run_type="chain", inputs={"requirement": requirement}, metadata={"run_id": run_id})`;
  - yields the root's URL (`rt.get_url()`, with a failure giving `None`);
  - otherwise yields `None` and does nothing.
  - LangGraph's own runs nest under it through LangSmith context propagation.
- `span_metadata(agent: str, calls: list[CallRecord], tools: list[ToolRecord], duration_ms: float) -> dict`:
  - returns `{"agent_name": agent, "token_count": Σ(prompt + completion), "tool_calls": len(tools), "tool_names": sorted({t.tool for t in tools}), "duration_ms": round(duration_ms, 1)}`.
- `agent_node(agent: Literal["pm", "coder", "qa"], fn: Callable[[ProjectState], dict]) -> Callable[[ProjectState], dict]`:
  - `functools.wraps(fn)`, keeping the `(state)` signature;
  - runs `fn` inside `agent_scope(agent)`;
  - snapshots `len(ctx.calls)` / `len(ctx.tools)` before, and computes `span_metadata` from the new records after, **also on exception** (`finally`);
  - if tracing is on and `langsmith.run_helpers.get_current_run_tree()` returns the node's run, `rt.metadata.update(meta)` (or `rt.add_metadata(meta)`) before it closes;
  - logs `agent_end` with `meta`.
  - Fallback if the call returns `None` under tracing: wrap with `traceable(name=agent)` and set metadata on that span. Verify which one works in Step 6.
- Graph wiring:
  - `pm_node` and `consolidate_node` → `"pm"`;
  - `coder_node` and `reflect_node` → `"coder"`;
  - `qa_node` → `"qa"`;
  - `dispatch_node` and `finalize` stay unwrapped (orchestration, not agents).

- [ ] **Step 1: Write the failing tests.**

```python
def test_span_metadata_fields():
    calls = [CallRecord(..., prompt_tokens=100, completion_tokens=20, ...), CallRecord(..., prompt_tokens=5, completion_tokens=5, ...)]
    tools = [ToolRecord(ts="t", agent="coder", tool="write_file", ok=True, duration_ms=1, attempts=1),
             ToolRecord(ts="t", agent="coder", tool="exec_python", ok=True, duration_ms=1, attempts=1)]
    assert span_metadata("coder", calls, tools, 12.34) == {"agent_name": "coder", "token_count": 130, "tool_calls": 2,
                                                         "tool_names": ["exec_python", "write_file"], "duration_ms": 12.3}

def test_tracing_off_is_noop(settings, monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False); monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    runner = scripted_runner(settings, passing_scripts()); runner.run(REQ)
    assert report_of(runner)["trace_url"] is None

def test_one_root_and_agent_child_spans(settings, monkeypatch):
    """Capture runs with a recording LangSmith client: exactly one root, all agent spans under its trace."""
    client = RecordingClient()  # test double: records create_run/update_run kwargs; see note below
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-test"); monkeypatch.setenv("LANGSMITH_TRACING", "true")
    with langsmith.run_helpers.tracing_context(enabled=True, client=client, project_name="t"):
        runner = scripted_runner(settings, passing_scripts()); runner.run(REQ)
    client.flush()
    roots = [r for r in client.runs.values() if r.get("parent_run_id") is None]
    assert len(roots) == 1 and roots[0]["name"].startswith("team-run:")
    agents = [r for r in client.runs.values() if (r.get("extra", {}).get("metadata") or {}).get("agent_name")]
    assert {r["extra"]["metadata"]["agent_name"] for r in agents} == {"pm", "coder", "qa"}
    for r in agents:
        md = r["extra"]["metadata"]
        assert {"token_count", "tool_calls", "duration_ms"} <= md.keys()
        assert r["trace_id"] == roots[0]["trace_id"]
```

  `RecordingClient`: build it as `langsmith.Client(api_key="test", auto_batch_tracing=False)` with `create_run` / `update_run` monkeypatched to merge kwargs into `self.runs[id]`. If the installed langsmith 0.14 posts through other methods, adapt the double to them. Keep the assertions unchanged.

- [ ] **Step 2: Run the tests.**
  - Run: `uv run pytest tests/test_tracing.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `observability/tracing.py`. Wire the nodes and replace the runner's stand-in.
- [ ] **Step 4: Run the tests and the suite.**
  - Run: `uv run pytest tests/test_tracing.py -v && uv run pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(tracing): one LangSmith root per run, agent spans with agent_name/token_count/tool_calls/duration_ms`.
- [ ] **Step 6: Live check (Brandon, about $0.05).**
  - Ask before running, since it costs money:

    ```
    op run --env-file=.env.op -- env LANGSMITH_TRACING=true LANGSMITH_PROJECT=multi-agent-dev-team uv run team "Write hello.py that prints hello" --qa-max-iterations 1
    ```

  - Open the logged trace URL.
  - Expected: one `team-run:<id>` root, with pm/coder/qa node spans carrying the four metadata fields in the UI.
  - If the metadata sits on extra `traceable` children instead of the node spans, that's acceptable. Note it in docs.

### Task 13: Validation audit and the invalid-output probe

**Files:**
- Modify: `agents/pm_agent.py` (consolidate fallback), `agents/qa_agent.py` (fix-plan and audit fallbacks)
- Create: `docs/evidence/v4/validation-audit.md`
- Test: `tests/test_security.py`

**Interfaces:**
- Silent fallbacks log WARNING events, with fields `error_class` and `message` (redacted):
  - `pm_consolidation_fallback` in `PMAgent.consolidate`;
  - `qa_fix_plan_fallback` in `QAAgent._fixes`;
  - `qa_audit_fallback` in `QAAgent._check_tests`.

- [ ] **Step 1: Write the failing tests** in `tests/test_security.py`.
  - Spy on the side-effect functions: monkeypatch `tools.dispatch.write_file`, `tools.dispatch.exec_python`, `agents.qa_agent.write_file` and `agents.qa_agent.exec_python` with recorders that call through.

```python
def test_invalid_pm_spec_never_reaches_tools(settings, spies):
    runner = scripted_runner(settings, ([ai("not a spec"), ai("still not")], [], []))
    state = runner.run(REQ)
    assert state["status"] == "failed" and "invalid spec twice" in state["error"] and spies.calls == []

def test_invalid_qa_suite_never_executes(settings, spies): ...
    # Coder delivers + reflects (scripts from tests/scenarios.py); QA returns two invalid suites
    # → no exec_python call from QA; the task closes with qa_outcome "qa_error"
def test_invalid_tool_call_rejected_before_write(settings, spies): ...
    # Coder calls write_file(path="../escape.py", content="x") and then write_file without content
    # → no write outside the workspace; observations start with "error:"
def test_consolidation_fallback_logs_warning(settings, caplog): ...  # RuntimeError("no merge") → "pm_consolidation_fallback" WARNING
def test_fix_plan_and_audit_fallbacks_log_warning(settings, caplog): ...
```

- [ ] **Step 2: Run the tests.**
  - Run: `uv run pytest tests/test_security.py -v`. Expected: the fallback-warning tests FAIL; the probe tests may already pass.
- [ ] **Step 3: Add the three `log_event(..., level=WARNING)` calls.**
- [ ] **Step 4: Write `docs/evidence/v4/validation-audit.md`.** It's a table with one row per item from §5.7:
  - `extract_plan` regex: display only, never reaches a tool;
  - fix strings interpolated into prompts: model input only;
  - `TestVerdict.corrected`: spliced only if it's exactly one function of that name (`_replace_tests`) and `QATestSuite.check()` passes, otherwise ignored;
  - the three fallbacks: now WARNING-logged;
  - every `with_structured_output` call: Pydantic-parsed before use;
  - tool args: `ARG_MODELS`.
  Each row cites file:line and the test proving it.
- [ ] **Step 5: Run the tests and commit.**
  - Run: `uv run pytest tests/test_security.py -v`. Expected: PASS.
  - Commit `feat(security): invalid-output probe tests, WARNING on silent fallbacks, validation audit`.

### Task 14: BST demo, supplied fixture, demo_check, mocked and live E2E

**Files:**
- Create: `tests/fixtures/bst_fixture.py`, `tests/test_e2e_bst.py`
- Modify: `app/team_cli.py`, `app/demo_check.py`, `tests/scenarios.py`, `tests/live/test_team_demo.py`
- Test: `tests/test_e2e_bst.py`, `tests/test_demo_check.py`

**Interfaces:**
- `DEMO_BST` in `app/team_cli.py`:
  - the published sentence verbatim: `"Build a Python module that implements a binary search tree with insert, search, and in-order traversal methods, with full test coverage."`;
  - then `"\n\nInterface (the supplied fixture's contract): "`;
  - then: module `bst.py` at the workspace root; class `BinarySearchTree` with `insert(value) -> None`, `search(value) -> bool`, `inorder() -> list` returning values in ascending order; duplicate inserts are ignored; values are mutually comparable.
- `DEMOS["bst"] = DEMO_BST`. `--demo bst` passes `demo_check=check_bst`. The exit code needs the check to pass, like `linked-list`. The check is titled "BST fixture check".
- `tests/fixtures/bst_fixture.py`: plain pytest functions that `from bst import BinarySearchTree`:
  - `test_insert_and_search_hit_and_miss`;
  - `test_inorder_sorted_random` (seeded `random.Random(4)`, 200 values);
  - `test_inorder_sorted_degenerate` (`range(300)` inserted ascending);
  - `test_duplicates_ignored`;
  - `test_empty_tree` (`search(1) is False`, `inorder() == []`).
- `check_bst(workspace: Path, state: dict) -> tuple[bool, str]` in `app/demo_check.py`. Lines:
  1. `_approved_line(state)` (reused).
  2. `FIXTURE_PATH` (resolved relative to the repo: `Path(__file__).parents[1] / "tests/fixtures/bst_fixture.py"`) run with `subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(FIXTURE_PATH)], cwd=workspace, env={**child_env(tmp), "PYTHONPATH": str(workspace)}, timeout=60)`. PASS/FAIL comes from the exit code, with the output tail.
  3. `coverage of bst.py under QA's tests: N% (reported, not gated)`: run the run's canonical QA suites with `--cov=bst --cov-report=term`, and parse the `bst.py` line's percentage. If it can't be measured, write `n/a (<reason>)`.
  - Passed = lines 1 and 2.
- `tests/scenarios.py` gains:
  - `BST_SRC`: a correct iterative-insert BST, so 300 degenerate inserts can't hit the recursion limit;
  - `CODER_BST_TESTS`, `QA_BST_TESTS`, `QA_BST_MANIFEST`;
  - `bst_spec() -> Spec` (one task, T1, with criteria for insert, search hit/miss, in-order, duplicates);
  - `bst_scripts() -> tuple[list, list, list]`: deliver, reflect (docstring revision), QA suite, approved round 1. Mirror `passing_scripts`.

- [ ] **Step 1: Write the fixture and its self-test.**
  - Write the fixture.
  - Add `tests/test_demo_check.py::test_bst_fixture_passes_reference_and_fails_broken`. It writes `BST_SRC` into a tmp workspace and `check_bst`'s fixture line is PASS. Then it writes a BST whose `inorder` returns insertion order, and the fixture line is FAIL.
- [ ] **Step 2: Write the mocked full-pipeline E2E** (`tests/test_e2e_bst.py`).

```python
def test_mocked_pipeline_builds_bst_that_passes_supplied_fixture(settings):
    """Real team graph, tools, exec_python, QA pytest run and supplied fixture; only the LLM is scripted."""
    runner = scripted_runner(settings, bst_scripts())
    final = None
    for ev in runner.stream(DEMO_BST, demo_check=check_bst):
        if ev.kind == "final":
            final = ev.data
    assert final["state"]["status"] == "completed"
    assert final["demo_check"]["passed"], final["demo_check"]["report"]
    assert "PASS supplied BST fixture" in final["demo_check"]["report"]
    assert "coverage of bst.py" in final["demo_check"]["report"]
    assert (settings.workspace_dir / "bst.py").is_file()
    assert json.loads(runner.cost_report_path.read_text())["outcome"] == "success"
```

- [ ] **Step 3: Run the tests.**
  - Run: `uv run pytest tests/test_e2e_bst.py tests/test_demo_check.py -v`. Expected: FAIL.
- [ ] **Step 4: Implement** `DEMO_BST`, `check_bst` and the scenario scripts.
- [ ] **Step 5: Add the live E2E** `test_bst_demo_passes_supplied_fixture(live_settings)` to `tests/live/test_team_demo.py`.
  - `runner.reset_memory()`, then stream `DEMO_BST` with `demo_check=check_bst`.
  - Assert `status == "completed"` and the demo check passed.
  - Also run the fixture directly (`subprocess` pytest) and assert exit 0.
- [ ] **Step 6: Run and commit.**
  - Run: `uv run pytest -q` (all pass) and `uv run pytest -m live --collect-only -q | grep bst` (collected).
  - Commit `feat(demo): BST grading demo with supplied fixture, demo_check, mocked and live E2E`.

### Task 15: Controlled-run evidence for retry, breaker, cache, invalid output

**Files:**
- Create: `scripts/resilience_runs.py`, `docs/evidence/v4/controlled/…`
- Test: `tests/test_controlled_runs.py` (extend)

**Interfaces:**
- `uv run python -m scripts.resilience_runs [--out docs/evidence/v4/controlled]` writes one dir per scenario. Each dir holds `events.jsonl` (the captured JSON log) and a `summary.md`, written the way v3's `_scrub` does it.
- Scenarios, all with scripted models and no keys:
  - `retry-then-success`: one 429, then success.
  - `retry-exhausted`: four 429s. Exactly 4 attempts, `retry_exhausted`, PM fails cleanly with a cost report.
  - `breaker`: threshold 3, cool-down 0.2 s, real sleep. Shows open, blocked, half_open, probe ok → closed; then a second cycle with probe fail → open. Every transition is a WARNING line.
  - `cache`: TTL 2 s. A repeated PM spec is a `cache_hit`; after `time.sleep(2.1)` it's `cache_expired`.
  - `invalid-output`: the Task 13 PM and QA probes, showing no tool events.
- `main(argv) -> int` returns 0 when every scenario's assertions hold.

- [ ] **Step 1: Write the failing test.**
  - `tests/test_controlled_runs.py::test_resilience_runs_produce_expected_events(tmp_path)` calls `main(["--out", str(tmp_path)])` and asserts it returns 0.
  - For each scenario it asserts the events: attempt counts, the transition sequence `closed→open→half_open→closed` then `…→half_open→open`, and `cache_hit` then `cache_expired`.
- [ ] **Step 2: Run the test.**
  - Run: `uv run pytest tests/test_controlled_runs.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `scripts/resilience_runs.py`.
- [ ] **Step 4: Run, generate the evidence, and commit.**
  - Run `uv run pytest tests/test_controlled_runs.py -v` (PASS), then `uv run python -m scripts.resilience_runs`.
  - Commit the script, the test and `docs/evidence/v4/controlled/` with the message `feat(evidence): controlled runs for retry, breaker, cache and invalid-output probe`.

### Task 16: Docker: Python 3.11 image, Compose with Chroma health check

**Files:**
- Modify: `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.env.example`, `requirements.txt` (via `make requirements`)
- Test: manual verification commands, plus `tests/test_docs.py::test_compose_shape` (new)

**Interfaces:**
- **Dockerfile:**
  - `FROM python:3.11-slim`, keeping the non-root user;
  - `COPY requirements.txt .` then `RUN pip install --no-cache-dir -r requirements.txt`;
  - `COPY . .` then `RUN pip install --no-cache-dir --no-deps .`;
  - keeps the embedding-model warmup `RUN python -c …`;
  - `ENTRYPOINT ["team"]`, `CMD ["--demo", "bst", "--reset-memory"]`.
- **Compose `chroma` service:**
  - `image: chromadb/chroma:<tag matching the locked chromadb client, 1.5.x>`;
  - volume `chroma-data:/data`;
  - a `healthcheck` against `/api/v2/heartbeat` (`interval: 5s`, `retries: 20`, `start_period: 10s`). Use the image's own tool, or `bash -c 'exec 3<>/dev/tcp/localhost/8000'`-style if there's no curl or python. Verify in Step 2.
- **Compose `team` service:**
  - `depends_on: {chroma: {condition: service_healthy}}`;
  - `env_file: [{path: .env, required: false}]`;
  - `environment` passthrough for keys plus `CHROMA_HOST: chroma`, `CHROMA_PORT: "8000"`;
  - volumes `./workspace:/app/workspace`, `./.coder:/app/.coder`, `./docs/cost_reports:/app/docs/cost_reports`;
  - no `restart`.
- **Other services:** `web` (profile `web`) and `coder` (profile `cli`) stay, both pointed at the Chroma service.
- **`.dockerignore`** adds `docs/cost_reports`, `docs/evidence`, `.env.*` except `.env.example`.

- [ ] **Step 1: Write the failing test** `tests/test_docs.py::test_compose_shape`. Parse `docker-compose.yml` with `yaml.safe_load` (PyYAML comes with chromadb's deps; if not, use `uv add --dev pyyaml`) and assert:
  - `team.depends_on.chroma.condition == "service_healthy"`;
  - `"healthcheck" in chroma`;
  - `team.env_file` is set;
  - `web` and `coder` have profiles;
  - the Dockerfile contains `FROM python:3.11-slim`, `-r requirements.txt` and `ENTRYPOINT ["team"]`.
  Run it and expect FAIL.
- [ ] **Step 2: Write the Dockerfile and Compose file**, then confirm the test passes.
  - Pick the Chroma tag: `rg -A1 'name = "chromadb"' uv.lock`, then use the matching `chromadb/chroma` tag.
  - Verify the health check works: `docker compose up -d chroma && docker compose ps` should show `healthy` within 60 s.
  - Run `uv run pytest tests/test_docs.py -v` (PASS).
- [ ] **Step 3: Verify secret precedence** with dummy values, no real keys.

```bash
printf 'ANTHROPIC_API_KEY=from-dotenv\n' > .env
env -u ANTHROPIC_API_KEY docker compose config --format json | jq -r '.services.team.environment.ANTHROPIC_API_KEY'   # expect from-dotenv
ANTHROPIC_API_KEY=from-host docker compose config --format json | jq -r '.services.team.environment.ANTHROPIC_API_KEY'  # expect from-host
rm .env; ANTHROPIC_API_KEY=from-host docker compose config --format json | jq -r '.services.team.environment.ANTHROPIC_API_KEY'  # expect from-host
```

  Choose the passthrough syntax that gives all three results. If none does, drop `environment:` for keys and document `op run --env-file=.env.op -- docker compose --env-file /dev/stdin …` as an alternative. Whatever you choose, never write a plaintext `.env` for Brandon's path.
- [ ] **Step 4: Check the image has no secrets.**
  - Run: `docker compose build team && docker run --rm --entrypoint sh multi-agent-dev-team:latest -c 'ls -a /app; env | grep -i key || true'`.
  - Expected: no `.env`, no key variables.
- [ ] **Step 5: Run a key-less smoke test.**
  - Run: `docker compose down -v && docker compose up --build --abort-on-container-exit team`, with `ANTHROPIC_API_KEY` unset.
  - Expected: Chroma reports healthy before `team` starts, and team exits 2 with `team: cannot run: ANTHROPIC_API_KEY is not set`.
- [ ] **Step 6: Commit** `feat(docker): Python 3.11 image from requirements.txt, ENTRYPOINT team, Compose Chroma service with health check`.

### Task 17: Docs: README v4, architecture, usage, evidence map

**Files:**
- Modify: `README.md`, `docs/architecture.md`, `docs/usage.md`
- Create: `docs/evidence/v4/README.md`
- Test: `tests/test_docs.py` (extend)

**Interfaces:**
- README v4 section headings, which the tests assert:
  - `## Production v4`
  - `### Quickstart (Docker)`
  - `### Quickstart (local, 1Password)`
  - `### Configuration`
  - `### Callable interface`
  - `### Running the tests`
  - `### Traces, logs and cost reports`
  - `### Reading the outcome`
  - `### Test map`
- **Test map table:** rubric item → test file:
  - PM unit → `test_pm_agent.py`;
  - Coder unit → `test_coder_agent.py`;
  - QA unit → `test_qa_agent.py`;
  - PM→Coder → `test_team_graph.py`;
  - Coder→QA termination → `test_team_loop.py`;
  - E2E mocked → `test_e2e_bst.py`;
  - E2E live → `tests/live/test_team_demo.py::test_bst_demo_passes_supplied_fixture`;
  - plus retry, breaker, cache, cost, tracing and security.
- **Reading the outcome:**
  - `team` exit 0 = `completed` and the demo check passed;
  - exit 1 = `partial` (an unresolved-issue report per task) or `failed`;
  - exit 2 = configuration, busy workspace, or Chroma unreachable;
  - exit 130 = interrupted;
  - the cost report's `outcome` matches.
- **Callable interface:** every `team` flag (`--demo api|large|linked-list|bst`, `--model`, `--pm-model`, `--qa-model`, …), `TeamRunner(settings).stream/run`, `make_model(settings, role)`, the settings table (env var, default) and the cost report JSON schema.

- [ ] **Step 1: Write the failing tests** in `tests/test_docs.py`.
  - `test_readme_v4_sections` checks the headings.
  - `test_readme_env_vars_match_settings` checks every `_ENV_VARS` value appears in README's configuration table.
  - `test_readme_demo_flags_match_cli` checks every `DEMOS` key appears.
  - `test_evidence_map_covers_c4` checks `docs/evidence/v4/README.md` mentions `C4.1` … `C4.9` and that every relative link in it exists.
  - Run them and expect FAIL.
- [ ] **Step 2: Write the docs.**
  - Add the resilience/observability diagram as a separate block. The existing team-graph diagram block stays identical, so `test_readme_diagram_matches_graph` still passes.
  - Document the v4 changes: `CODER_MODEL` is Coder-only with a Haiku default; the response cache survives `--reset-memory`; the `.coder` bind mount keeps the `reset_memory` sync.
- [ ] **Step 3: Write the evidence map,** C4.1–C4.9 → evidence file plus test.
  - Mark the live items "pending Task 18" until they exist.
- [ ] **Step 4: Run the tests and commit.**
  - Run: `uv run pytest tests/test_docs.py -v`. Expected: PASS.
  - Commit `docs: v4 README, architecture, usage and C4 evidence map`.

### Task 18: Live evidence and final verification (needs Brandon's keys; paid calls)

**Files:**
- Create: `docs/evidence/v4/live-bst/…`, `docs/evidence/v4/docker-clean-start.log`, `docs/evidence/v4/mocked-suite.txt`
- Modify: `README.md` and `docs/evidence/v4/README.md` (public trace link, remove "pending")

- [ ] **Step 1: Run the mocked suite as evidence.**
  - Run: `uv run pytest -q 2>&1 | tail -5 | tee docs/evidence/v4/mocked-suite.txt`.
  - Expected: `N passed`, 0 failed, live tests deselected.
- [ ] **Step 2: Run the live BST demo with tracing.**
  - Ask Brandon first; this is a paid run.
  - Run: `op run --env-file=.env.op -- env LANGSMITH_TRACING=true uv run team --demo bst --reset-memory`.
  - Expected: exit 0 and "BST fixture check: PASSED".
  - Copy `final-state.json`, `events.jsonl`, `cost-report.json` and `demo-check.txt` into `docs/evidence/v4/live-bst/`.
- [ ] **Step 3: Brandon publishes the root trace and shares the link.**
  - In LangSmith, Brandon creates a public share link for the `team-run:<id>` root.
  - Confirm the trace shows no key values.
  - Add the link to the README and the evidence map.
- [ ] **Step 4: Clean-start Docker run.**
  - Run: `docker compose down -v --remove-orphans && rip workspace .coder; op run --env-file=.env.op -- docker compose up --build 2>&1 | tee docs/evidence/v4/docker-clean-start.log`.
  - Expected: Chroma `healthy` before the team logs `run_start`, a run outcome line, and a new `docs/cost_reports/<id>.json` on the host.
- [ ] **Step 5: Run the live tests.**
  - Run: `op run --env-file=.env.op -- make verify`.
  - Expected: the live tests pass, including `test_bst_demo_passes_supplied_fixture`.
- [ ] **Step 6: Commit the evidence** with the message `docs(evidence): v4 live BST run, public trace, Docker clean start, mocked suite`.

---

## Spec coverage check

| Spec | Task |
|---|---|
| §5.1 3.11 | 1 |
| §5.2 tracing | 12 |
| §5.2 structured logs | 5, 11 |
| §5.2 public trace link | 18 |
| §5.3 retry | 3, 8, 9 |
| §5.3 idempotent writes | 9 |
| §5.3 SDK retries 0 | 8 |
| §5.4 breakers | 4, 8, 9 |
| §5.5 cost records and report | 6, 8, 11 |
| §5.5 aborted runs | 11 |
| §5.5 Docker path | 16 |
| §5.6 cache | 7, 8 |
| §5.6 prompt caching | 8 |
| §5.6 models per role | 2, 8 |
| §5.7 env allowlist | 10 |
| §5.7 probe and audit | 13 |
| §5.7 QA tool args | 9 |
| §5.7 secrets and .dockerignore | 16 |
| §5.8 mocked by default | 1 |
| §5.8 new unit tests | 3–13 |
| §5.8 BST contract, E2E and coverage | 14 |
| §5.8 controlled runs | 15 |
| §5.9 Docker | 16 |
| §5.10 docs | 17 |
| §6 evidence | 13, 15, 18 |
