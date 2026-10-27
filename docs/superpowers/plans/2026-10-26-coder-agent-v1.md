# Coder Agent v1.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A LangGraph Coder agent (Module 1 of the Saras AI capstone) that meets rubric C1.1–C1.6, with a rich terminal walkthrough and a read-only "mini-IDE" web UI.

**Architecture:** A hand-built `StateGraph` (retrieve_memory → agent → tools → … → finalize) drives Claude Sonnet through ≤10 tool-calling iterations against three confined tools. A subprocess runner records exactly which workspace files each execution loaded, and `finalize` (no model call) builds a Pydantic `CoderResult` from disk and from those execution records. `CoderAgent` wraps the graph with a cross-process workspace lock, two memory layers (sliding window + Chroma), and a step-event stream that the CLI and web UI both consume.

**Tech Stack:** Python 3.12, uv, LangGraph 1.x, `langchain` (`init_chat_model`) + `langchain-anthropic`, chromadb (`PersistentClient`, default local embedding), Pydantic v2, filelock, rich, FastAPI + uvicorn, pytest.

**Spec:** `docs/superpowers/specs/2026-10-25-coder-agent-v1-design.md`. Read it in full before Task 1. Section numbers below (§6.2 etc.) refer to it.

## Global Constraints

- Repo root: `/Volumes/Tukar/Code/ai-certifications/SarasAI-Masters-AI-Eng/multi-agent-dev-team`. Flat layout: `agents/ tools/ memory/ orchestration/ app/ web/ tests/ docs/` are top-level packages and dirs (no `src/`).
- Required top-level files (C1.1): `.env.example`, `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `README.md`, plus `agents/pm_agent.py` and `agents/qa_agent.py` placeholders.
- Python ≥ 3.12, managed by uv (`pyproject.toml` + `uv.lock`). Add dependencies with `uv add`. **Never `uv pip`**: a local hook blocks it. Regenerate `requirements.txt` with `uv export --no-hashes --no-emit-project --format requirements-txt > requirements.txt` whenever dependencies change.
- This Mac's shell aliases `rm`→rip and `find`→fd, so use `/bin/rm` and `/usr/bin/find` in commands. A bare `python` outside a venv is a wrapper that errors, so always use `uv run …` or `sys.executable`.
- Default model `anthropic:claude-sonnet-5-5`, via `CODER_MODEL`. Never hard-code a key. `.env.op` holds `op://` references and is run as `op run --env-file=.env.op -- <cmd>`.
- `MAX_ITERATIONS` cap = **10**. Configurable down, never above. One iteration = one model call. `finalize` makes no model call.
- `exec_python` default timeout **10 s** (`EXEC_TIMEOUT_SECONDS`). Never `eval`/`exec` generated code in the agent process. Never run with `-O`, `-I` or `-P`.
- Never commit `.env`, `workspace/`, `.chroma/` or `.coder/`.
- Check LangGraph / langchain / chromadb call shapes against live docs (LangChain docs MCP or context7) before relying on a signature this plan names. Where the docs disagree, keep this plan's *behaviour* and adapt the call.
- Commit after every task. Messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do **not** create the GitHub repo or push before Task 14.

## Review Focus

The spec doesn't cover these inputs, but they're likely to hit a real user. Each has a test in the task that owns the code.

1. **Paths prefixed `workspace/`.** The model writes `workspace/palindrome.py` (this happened in Lab 1.3). Expected: same file as `palindrome.py`, not `workspace/workspace/…`. Test in Task 2.
2. **Output floods.** A script prints in a tight loop. Expected: the timeout still fires, and captured output is truncated (no memory blow-up, no pipe deadlock). Test in Task 4.
3. **Scripts that read stdin.** A script calls `input()`. Expected: immediate `EOFError` (stdin is `/dev/null`), not a 10 s hang. Test in Task 4.
4. **Non-UTF-8 bytes.** A binary file is read, or a program prints invalid UTF-8. Expected: decoded with replacement characters, no crash. Tests in Tasks 3 and 4.
5. **Model or provider failure mid-run.** A 401, a 402 (no credit) or a network error. Expected: a `CoderResult` with `status="error"` and a readable message, the lock released, and a CLI exit code of 1 with no traceback. Tests in Tasks 9 and 10.

---

## File Structure

```
agents/__init__.py
agents/config.py         Settings (env → validated config, path layout)
agents/model.py          make_model(), provider_prerequisite()
agents/prompts.py        SYSTEM_PROMPT
agents/events.py         StepEvent
agents/result.py         ExecutionResult, CoderResult (+ consistency validator)
agents/coder_agent.py    CoderAgent: lock, generation check, graph run, memory write-back
agents/pm_agent.py       placeholder (docstring only)
agents/qa_agent.py       placeholder (docstring only)
tools/__init__.py
tools/workspace.py       ToolError, WorkspaceError, WorkspaceBusyError, Workspace, RunLock, ResetGeneration, sha256_file
tools/file_tools.py      read_file, write_file, WriteOutcome
tools/exec_python.py     exec_python, ExecOutcome (Popen + process-group kill)
tools/_runner.py         subprocess entry: runs target, writes execution record JSON
tools/specs.py           arg models, TOOL_SPECS, ACTION_TOOLS, FINISH
tools/dispatch.py        ToolContext, ActionOutcome, run_action
memory/__init__.py
memory/window.py         Turn, SlidingWindow
memory/semantic.py       MemoryHit, SemanticMemory (Chroma)
memory/context.py        format_memory_context
orchestration/__init__.py
orchestration/state.py   CoderState, WriteInfo, ExecInfo
orchestration/routing.py Decision, classify, extract_plan
orchestration/verify.py  Verification, verify_delivery
orchestration/finalize.py build_result
orchestration/graph.py   GraphDeps, build_graph, initial_state
app/__init__.py
app/render.py            rich renderers for StepEvent / CoderResult
app/cli.py               `coder` entry point
app/web.py               `coder-web` FastAPI app
web/index.html, web/app.js, web/style.css
tests/fakes.py           ScriptedChatModel, tool_call(), FakeEmbedding
tests/conftest.py        settings fixture (tmp project dir), live gating
tests/test_*.py          unit tests per module
tests/live/test_smoke.py live smoke + memory follow-up
docs/architecture.md, docs/usage.md, docs/evidence/
```

Runtime layout under the project dir (all gitignored): `workspace/` (agent files), `.chroma/` (Chroma), and `.coder/` (`workspace.lock`, `generation`, `exec/` execution records). The spec puts the lock "in the workspace parent". `.coder/` sits in that parent, which keeps the root tidy.

---

### Task 1: Scaffold, settings, and first commit

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `.env.op`, `README.md` (stub), `Makefile`, `requirements.txt`, all package `__init__.py` files, `agents/pm_agent.py`, `agents/qa_agent.py`, `agents/config.py`, `tests/conftest.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings` (pydantic `BaseModel`) with fields `project_dir: Path`, `model: str = "anthropic:claude-sonnet-5-5"`, `max_iterations: int = 10`, `exec_timeout_s: float = 10.0`, `window_turns: int = 10`, `memory_top_k: int = 3`, `memory_max_distance: float = 0.65`, `max_tokens: int = 8192`. Properties `workspace_dir` (`project_dir/"workspace"`), `chroma_dir` (`project_dir/".chroma"`), `state_dir` (`project_dir/".coder"`). `Settings.from_env(**overrides) -> Settings` reads `CODER_PROJECT_DIR` (default cwd), `CODER_MODEL`, `CODER_MAX_ITERATIONS`, `EXEC_TIMEOUT_SECONDS`, `MEMORY_WINDOW_TURNS`, `MEMORY_TOP_K`, `MEMORY_MAX_DISTANCE`, `CODER_MAX_TOKENS`; overrides that are `None` are ignored. `MAX_ITERATIONS_CAP = 10`.
- Produces: fixture `settings(tmp_path) -> Settings` with `project_dir=tmp_path` and `exec_timeout_s=5`.

- [ ] **Step 1: Initialise the repo**

`git init -b main`. `uv init --bare --python 3.12`. Use hatchling as the build backend with `[tool.hatch.build.targets.wheel] packages = ["agents","tools","memory","orchestration","app"]`. Add the scripts `coder = "app.cli:main"` and `coder-web = "app.web:main"`.

Install with `uv add langgraph langchain langchain-anthropic chromadb pydantic python-dotenv filelock rich fastapi uvicorn`, then `uv add --dev pytest httpx`. Add the optional extras `openai = ["langchain-openai"]` and `ollama = ["langchain-ollama"]`.

Pytest config in `pyproject.toml`: `testpaths = ["tests"]`, `markers = ["live: calls a real model provider"]`, `addopts = "-p no:cacheprovider"`.

`.gitignore`: `.venv/ __pycache__/ .env workspace/ .chroma/ .coder/ .pytest_cache/`.

- [ ] **Step 2: Env templates**

`.env.example`:
- `ANTHROPIC_API_KEY=`
- `CODER_MODEL=anthropic:claude-sonnet-5-5`
- `EXEC_TIMEOUT_SECONDS=10`
- `CODER_MAX_ITERATIONS=10`
- `MEMORY_WINDOW_TURNS=10`
- commented `OPENAI_API_KEY=` / `OLLAMA_BASE_URL=` / `LANGSMITH_TRACING=` / `LANGSMITH_API_KEY=` / `LANGSMITH_PROJECT=`

`.env.op`: the same keys, with `ANTHROPIC_API_KEY=op://Private/Anthropic Academy - API Key/<field>` and `LANGSMITH_API_KEY=op://Private/LangSmith API key/credential`. Find `<field>` with `op item get "Anthropic Academy - API Key" --vault Private --format json` and **list field labels only; never print values**. If that is blocked or needs biometrics, ask Brandon for the field label.

- [ ] **Step 3: Write the failing test** `tests/test_config.py`

```python
def test_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("CODER_MODEL", raising=False)
    s = Settings.from_env(project_dir=tmp_path)
    assert s.model == "anthropic:claude-sonnet-5-5"
    assert s.max_iterations == 10 and s.exec_timeout_s == 10.0 and s.window_turns == 10
    assert s.workspace_dir == tmp_path / "workspace"
    assert s.state_dir == tmp_path / ".coder" and s.chroma_dir == tmp_path / ".chroma"

def test_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CODER_MAX_ITERATIONS", "3")
    monkeypatch.setenv("EXEC_TIMEOUT_SECONDS", "2.5")
    s = Settings.from_env(project_dir=tmp_path)
    assert (s.max_iterations, s.exec_timeout_s) == (3, 2.5)

@pytest.mark.parametrize("bad", [0, 11, 50])
def test_max_iterations_never_above_cap(tmp_path, bad):
    with pytest.raises(ValidationError, match="between 1 and 10"):
        Settings.from_env(project_dir=tmp_path, max_iterations=bad)

def test_timeout_must_be_positive(tmp_path):
    with pytest.raises(ValidationError):
        Settings.from_env(project_dir=tmp_path, exec_timeout_s=0)
```

- [ ] **Step 4: Run it and see it fail.** `uv run pytest tests/test_config.py -v` → FAIL (ImportError).

- [ ] **Step 5: Implement `agents/config.py`**

`from_env` calls `load_dotenv()` (no override) before reading the env.

The `max_iterations` validator error message must contain `between 1 and 10`. The CLI shows it verbatim, so it's the assessor-facing text.

- [ ] **Step 6: Placeholders and README stub**

- `agents/pm_agent.py` and `agents/qa_agent.py`: module docstrings describing their Module 2/3 roles, nothing else.
- `README.md`: title, a one-line pitch, and "Setup instructions: see below (in progress)". It gets finished in Task 13.
- `Makefile` with targets:
  - `setup` (`uv sync`)
  - `test` (`uv run pytest`)
  - `verify` (`REQUIRE_LIVE=1 uv run pytest -rs`)
  - `demo` (`uv run coder --demo --reset-memory`)
  - `web` (`uv run coder-web`)
  - `requirements` (the uv export command above)
- Generate `requirements.txt`.

- [ ] **Step 7: Run it and see it pass.** `uv run pytest -v` → 6 passed.

- [ ] **Step 8: Commit.** Include the spec and this plan.

```bash
git add -A && git commit -m "chore: scaffold Coder agent repo with settings"
```

---

### Task 2: Workspace confinement, run lock, reset generation

**Files:**
- Create: `tools/workspace.py`
- Test: `tests/test_workspace.py`

**Interfaces:**
- Produces:
  - `class ToolError(Exception)`; `class WorkspaceError(ToolError)`; `class WorkspaceBusyError(Exception)`. The busy error is not a ToolError: it is raised before a run starts.
  - `Workspace(root: Path)`:
    - creates `root` if it doesn't exist.
    - `.resolve(path: str) -> Path`: resolved absolute path inside root, else raises `WorkspaceError`.
    - `.relpath(p: Path) -> str`: POSIX-style path relative to root.
    - `.list_files() -> list[str]`: sorted relative paths of regular files, skipping any path component that starts with `.` or is `__pycache__`.
    - `.clear() -> None`: removes everything under root and keeps root.
  - `RunLock(state_dir: Path)`: a context manager wrapping `filelock.FileLock(state_dir/"workspace.lock")` with a non-blocking acquire. On contention it raises `WorkspaceBusyError("workspace busy: another Coder run is in progress")`. Also has `.is_locked_elsewhere() -> bool`, which tries to acquire and release.
  - `ResetGeneration(state_dir: Path)`: `.read() -> int` (0 if the file is missing) and `.bump() -> int`, using the file `state_dir/"generation"`.
  - `sha256_file(p: Path) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
def test_relative_and_absolute_inside(tmp_path):
    ws = Workspace(tmp_path / "workspace")
    assert ws.resolve("a/b.py") == (tmp_path / "workspace/a/b.py").resolve()
    assert ws.resolve(str(ws.root / "c.py")) == (ws.root / "c.py").resolve()

@pytest.mark.parametrize("bad", ["../x.py", "/etc/passwd", "a/../../x.py"])
def test_escape_rejected(tmp_path, bad):
    with pytest.raises(WorkspaceError):
        Workspace(tmp_path / "workspace").resolve(bad)

def test_symlink_escape_rejected(tmp_path):
    ws = Workspace(tmp_path / "workspace")
    (ws.root / "link").symlink_to(tmp_path)
    with pytest.raises(WorkspaceError):
        ws.resolve("link/outside.py")

def test_leading_workspace_prefix_is_stripped(tmp_path):        # Review Focus 1
    ws = Workspace(tmp_path / "workspace")
    assert ws.resolve("workspace/p.py") == ws.resolve("p.py")
    assert ws.resolve("./workspace/p.py") == ws.resolve("p.py")

def test_list_files_skips_hidden_and_pycache(tmp_path):
    ws = Workspace(tmp_path / "workspace")
    for rel in ["a.py", "pkg/b.py", ".coder_snippets/s.py", "__pycache__/a.pyc"]:
        (ws.root / rel).parent.mkdir(parents=True, exist_ok=True); (ws.root / rel).write_text("x")
    assert ws.list_files() == ["a.py", "pkg/b.py"]

def test_run_lock_is_exclusive_across_processes(tmp_path):
    holder = subprocess.Popen([sys.executable, "-c", HOLD_LOCK_SNIPPET, str(tmp_path)],
                              stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "locked"
    try:
        with pytest.raises(WorkspaceBusyError, match="workspace busy"):
            with RunLock(tmp_path):
                pass
    finally:
        holder.kill(); holder.wait()
    with RunLock(tmp_path):        # free again once the holder is gone
        pass

def test_generation_bumps(tmp_path):
    g = ResetGeneration(tmp_path)
    assert g.read() == 0 and g.bump() == 1 and ResetGeneration(tmp_path).read() == 1
```

`HOLD_LOCK_SNIPPET` is a test-module string. It imports `RunLock`, enters it on `sys.argv[1]`, prints `locked`, flushes, and sleeps for 30 s. Run it with `cwd` set to the repo root so `tools` is importable.

- [ ] **Step 2: Run it and see it fail.** `uv run pytest tests/test_workspace.py -v` → FAIL (ImportError).

- [ ] **Step 3: Implement `tools/workspace.py`**

`resolve` steps:
1. Strip one leading `workspace/` or `./workspace/` component from relative inputs.
2. Join the result onto root.
3. Call `Path.resolve()`.
4. Require `resolved == root_resolved or root_resolved in resolved.parents`.

The error message names the offending path and the workspace root.

- [ ] **Step 4: Run it and see it pass.** `uv run pytest tests/test_workspace.py -v` → 9 passed.

- [ ] **Step 5: Commit.**

```bash
git add tools/workspace.py tests/test_workspace.py && git commit -m "feat(tools): workspace confinement, run lock, reset generation"
```

---

### Task 3: File tools

**Files:**
- Create: `tools/file_tools.py`
- Test: `tests/test_file_tools.py`

**Interfaces:**
- Consumes: `Workspace`, `WorkspaceError`, `ToolError`, and `sha256_file` from Task 2.
- Produces:
  - `@dataclass(frozen=True) WriteOutcome(path: str, sha256: str, bytes: int, lines: int)`.
  - `write_file(ws: Workspace, path: str, content: str) -> WriteOutcome`: creates parent dirs and writes UTF-8.
  - `read_file(ws: Workspace, path: str, max_chars: int = 20_000) -> str`:
    - decodes UTF-8 with `errors="replace"`.
    - if truncated, appends `\n[truncated: N more characters]`.
    - raises `ToolError("file not found: <rel>")` for a missing file and `ToolError("not a file: <rel>")` for a directory.

- [ ] **Step 1: Write the failing tests**

```python
def test_write_then_read_roundtrip(tmp_path):
    ws = Workspace(tmp_path / "ws")
    out = write_file(ws, "pkg/m.py", "a = 1\nb = 2\n")
    assert (out.path, out.lines, out.bytes) == ("pkg/m.py", 2, 12)
    assert out.sha256 == hashlib.sha256(b"a = 1\nb = 2\n").hexdigest()
    assert read_file(ws, "pkg/m.py") == "a = 1\nb = 2\n"

def test_read_missing_is_tool_error(tmp_path):
    with pytest.raises(ToolError, match="file not found: nope.py"):
        read_file(Workspace(tmp_path / "ws"), "nope.py")

def test_read_truncates(tmp_path):
    ws = Workspace(tmp_path / "ws"); write_file(ws, "big.txt", "x" * 50)
    assert read_file(ws, "big.txt", max_chars=10) == "x" * 10 + "\n[truncated: 40 more characters]"

def test_read_binary_does_not_crash(tmp_path):                    # Review Focus 4
    ws = Workspace(tmp_path / "ws"); (ws.root / "b.bin").write_bytes(b"\xff\xfe\x00ok")
    assert "ok" in read_file(ws, "b.bin")

def test_write_outside_rejected(tmp_path):
    with pytest.raises(WorkspaceError):
        write_file(Workspace(tmp_path / "ws"), "../escape.py", "x")
```

- [ ] **Step 2: Run it and see it fail.** `uv run pytest tests/test_file_tools.py -v` → FAIL.
- [ ] **Step 3: Implement `tools/file_tools.py`.**
- [ ] **Step 4: Run it and see it pass.** → 5 passed.
- [ ] **Step 5: Commit.** `git commit -m "feat(tools): read_file and write_file"`

---

### Task 4: `exec_python` and the execution-record runner

**Files:**
- Create: `tools/exec_python.py`, `tools/_runner.py`
- Test: `tests/test_exec_python.py`

**Interfaces:**
- Consumes: `Workspace`, `ToolError`, and `sha256_file`.
- Produces:
  - `@dataclass(frozen=True) ExecOutcome` with fields:
    - `target: str` (relative to the workspace)
    - `mode: Literal["script","pytest"]`
    - `exit_code: int | None` (None when timed out)
    - `stdout: str`, `stderr: str`
    - `timed_out: bool`
    - `duration_s: float`
    - `loaded: dict[str, str]` (relative path → sha256 at load time)
  - `ExecOutcome.ok` (a property): `exit_code == 0 and not timed_out`.
  - `ExecOutcome.to_observation() -> str`, in the format: `exit_code=…` / `timed_out=…` / `duration=…s` / `--- stdout ---` / … / `--- stderr ---` / …
- Produces: `exec_python(ws: Workspace, *, record_dir: Path, path: str | None = None, code: str | None = None, timeout: float = 10.0, mode: Literal["auto","script","pytest"] = "auto") -> ExecOutcome`.
  - Exactly one of `path` or `code` must be given, else `ToolError`.
  - `code` is first written to `ws.root/".coder_snippets"/f"snippet_{n}.py"`, where n increments per workspace.
  - A missing target raises `ToolError`.
  - `"auto"` means pytest when the target name matches `test_*.py` or `*_test.py`, else script.
- Constants: `MAX_OUTPUT_CHARS = 8000` per stream. `DEFAULT_TIMEOUT_S = 10.0`.

**Execution contract (spec §6.2, which wins over this summary if they differ):**
- Command: `[sys.executable, "-s", <abs path of tools/_runner.py>, mode, <abs target>, <abs workspace root>, <abs record json>]`, with `cwd=ws.root` and `stdin=subprocess.DEVNULL`.
- stdout and stderr go to temp files under `record_dir`, not pipes. This avoids pipe deadlock; afterwards read the first `MAX_OUTPUT_CHARS` with `errors="replace"`, plus a `[truncated]` marker.
- POSIX: `start_new_session=True`. On timeout, `os.killpg(proc.pid, SIGKILL)` and then `proc.wait()`. Windows: `creationflags=CREATE_NEW_PROCESS_GROUP`, and on timeout `taskkill /T /F /PID`.
- `loaded` is read from the record JSON. A missing or unreadable record gives `{}`.

**`tools/_runner.py` (runs inside the child; standard library only, except pytest in pytest mode):**
1. Replace `sys.path[0]`, which is the runner's own `tools/` dir, with the target's directory. This keeps `tools/` modules from shadowing workspace modules.
2. Record `sha256` of the target.
3. Insert a `MetaPathFinder` at `sys.meta_path[0]`. It delegates to `importlib.machinery.PathFinder.find_spec`. When the spec's `origin` resolves inside the workspace root, it records `sha256(origin)`, which is the load-time hash. Then it returns the spec unchanged.
4. Script mode: `runpy.run_path(target, run_name="__main__")`. Pytest mode: `sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", target]))`.
5. In `finally`, write `{"loaded": {relpath: sha}}` to the record path. Let `SystemExit` and exceptions propagate after the write, so the exit code and traceback are the real ones.

- [ ] **Step 1: Write the failing tests**

```python
def run(ws, tmp_path, **kw):
    return exec_python(ws, record_dir=tmp_path / "rec", timeout=kw.pop("timeout", 5), **kw)

def test_script_success_and_record(tmp_path):
    ws = Workspace(tmp_path / "ws"); write_file(ws, "hello.py", "print('hi')")
    out = run(ws, tmp_path, path="hello.py")
    assert out.ok and out.stdout.strip() == "hi" and out.mode == "script"
    assert out.loaded == {"hello.py": sha256_file(ws.root / "hello.py")}

def test_sibling_import_recorded(tmp_path):
    ws = Workspace(tmp_path / "ws")
    write_file(ws, "palindrome.py", "def is_pal(s): return s == s[::-1]")
    write_file(ws, "main.py", "from palindrome import is_pal\nprint(is_pal('aba'))")
    out = run(ws, tmp_path, path="main.py")
    assert out.ok and out.stdout.strip() == "True"
    assert set(out.loaded) == {"main.py", "palindrome.py"}

def test_pytest_mode_two_files(tmp_path):
    ws = Workspace(tmp_path / "ws")
    write_file(ws, "palindrome.py", "def is_pal(s): return s == s[::-1]")
    write_file(ws, "test_palindrome.py", "from palindrome import is_pal\ndef test_it(): assert is_pal('aba')")
    out = run(ws, tmp_path, path="test_palindrome.py")
    assert out.mode == "pytest" and out.ok and "1 passed" in out.stdout
    assert set(out.loaded) == {"test_palindrome.py", "palindrome.py"}

def test_failing_assertion_nonzero(tmp_path):
    ws = Workspace(tmp_path / "ws"); write_file(ws, "test_bad.py", "def test_x(): assert 1 == 2")
    out = run(ws, tmp_path, path="test_bad.py")
    assert out.exit_code not in (0, None) and not out.ok

def test_infinite_loop_killed(tmp_path):
    ws = Workspace(tmp_path / "ws"); write_file(ws, "loop.py", "while True: pass")
    t0 = time.monotonic(); out = run(ws, tmp_path, path="loop.py", timeout=1)
    assert out.timed_out and out.exit_code is None and time.monotonic() - t0 < 4

def test_spawned_child_killed_too(tmp_path):
    ws = Workspace(tmp_path / "ws")
    write_file(ws, "spawn.py", SPAWN_SNIPPET)   # starts `sleep 60`-equivalent python child, writes its pid to child.pid, then loops
    out = run(ws, tmp_path, path="spawn.py", timeout=1)
    pid = int((ws.root / "child.pid").read_text())
    assert out.timed_out
    time.sleep(0.3)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)

def test_output_flood_truncated_and_timed_out(tmp_path):         # Review Focus 2
    ws = Workspace(tmp_path / "ws"); write_file(ws, "flood.py", "while True: print('x' * 1000)")
    out = run(ws, tmp_path, path="flood.py", timeout=1)
    assert out.timed_out and len(out.stdout) <= MAX_OUTPUT_CHARS + 50 and "[truncated]" in out.stdout

def test_stdin_is_devnull(tmp_path):                              # Review Focus 3
    ws = Workspace(tmp_path / "ws"); write_file(ws, "ask.py", "input('name? ')")
    out = run(ws, tmp_path, path="ask.py", timeout=5)
    assert not out.timed_out and "EOFError" in out.stderr

def test_invalid_utf8_output(tmp_path):                           # Review Focus 4
    ws = Workspace(tmp_path / "ws"); write_file(ws, "b.py", "import sys; sys.stdout.buffer.write(b'\\xff ok')")
    assert "ok" in run(ws, tmp_path, path="b.py").stdout

def test_inline_code_written_to_workspace(tmp_path):
    ws = Workspace(tmp_path / "ws")
    out = run(ws, tmp_path, code="print(2 + 2)")
    assert out.ok and out.stdout.strip() == "4" and out.target.startswith(".coder_snippets/")
    assert (ws.root / out.target).exists()

def test_target_outside_rejected(tmp_path):
    with pytest.raises(WorkspaceError):
        run(Workspace(tmp_path / "ws"), tmp_path, path="../x.py")

def test_no_eval_in_tools():
    for f in Path("tools").glob("*.py"):
        assert not re.search(r"\b(eval|exec)\(", f.read_text()), f
```

This last test must not match `exec_python(`. The `\b` before `exec` handles that, because `_` is a word character.

- [ ] **Step 2: Run it and see it fail.** `uv run pytest tests/test_exec_python.py -v` → FAIL.
- [ ] **Step 3: Implement `tools/_runner.py` and `tools/exec_python.py`** as described in the contract above.
- [ ] **Step 4: Run it and see it pass.** → 12 passed, in under 20 s total.
- [ ] **Step 5: Commit.** `git commit -m "feat(tools): exec_python with process-group timeout and execution records"`

---

### Task 5: Tool specs and dispatch

**Files:**
- Create: `tools/specs.py`, `tools/dispatch.py`
- Test: `tests/test_dispatch.py`

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces, in `tools/specs.py`:
  - Pydantic arg models:
    - `ReadFileArgs(path: str)`
    - `WriteFileArgs(path: str, content: str)`
    - `ExecPythonArgs(path: str | None = None, code: str | None = None, timeout: float | None = None, mode: Literal["auto","script","pytest"] = "auto")`
    - `FinishArgs(explanation: str (min_length=1), plan: list[str] (min_length=1), entry_point: str | None = None)`
  - Constants: `FINISH = "finish"`, `ACTION_TOOLS = ("read_file", "write_file", "exec_python")`.
  - `TOOL_SPECS: list[dict]`: one `{"name", "description", "parameters": Model.model_json_schema()}` per tool, including `finish`. Each description says paths are relative to the workspace. The `finish` description says it must be called alone, after the last action.
- Produces, in `tools/dispatch.py`:
  - `@dataclass ToolContext(workspace: Workspace, exec_timeout_s: float, record_dir: Path)`.
  - `@dataclass ActionOutcome(observation: str, write: WriteOutcome | None = None, execution: ExecOutcome | None = None)`.
  - `run_action(ctx: ToolContext, name: str, args: dict) -> ActionOutcome`. It validates `args` with the matching model; a validation failure becomes `ToolError("invalid arguments for <name>: …")`. An unknown name becomes `ToolError("unknown tool: <name>")`. A timeout of `None` means `ctx.exec_timeout_s`.
  - Observations:
    - write: `"wrote <path> (<lines> lines, <bytes> bytes)"`
    - read: the file content
    - exec: `ExecOutcome.to_observation()`

- [ ] **Step 1: Write the failing tests**

```python
def ctx(tmp_path): return ToolContext(Workspace(tmp_path / "ws"), 5.0, tmp_path / "rec")

def test_specs_cover_all_tools():
    assert [s["name"] for s in TOOL_SPECS] == ["read_file", "write_file", "exec_python", "finish"]
    assert all(s["parameters"]["type"] == "object" for s in TOOL_SPECS)

def test_write_then_exec(tmp_path):
    c = ctx(tmp_path)
    w = run_action(c, "write_file", {"path": "a.py", "content": "print(1)"})
    assert w.write.path == "a.py" and w.observation.startswith("wrote a.py")
    e = run_action(c, "exec_python", {"path": "a.py"})
    assert e.execution.ok and "exit_code=0" in e.observation

def test_default_timeout_comes_from_context(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr("tools.dispatch.exec_python", lambda ws, **kw: seen.update(kw) or FAKE_OUTCOME)
    run_action(ctx(tmp_path), "exec_python", {"code": "print(1)"})
    assert seen["timeout"] == 5.0

@pytest.mark.parametrize("name,args,msg", [
    ("write_file", {"path": "a.py"}, "invalid arguments for write_file"),
    ("rm_rf", {}, "unknown tool: rm_rf"),
])
def test_bad_calls_are_tool_errors(tmp_path, name, args, msg):
    with pytest.raises(ToolError, match=msg):
        run_action(ctx(tmp_path), name, args)
```

- [ ] **Step 2: Run it and see it fail.** `uv run pytest tests/test_dispatch.py -v` → FAIL.
- [ ] **Step 3: Implement both modules.**
- [ ] **Step 4: Run it and see it pass.** → 5 passed.
- [ ] **Step 5: Commit.** `git commit -m "feat(tools): tool specs and dispatcher"`

---

### Task 6: Memory layers

**Files:**
- Create: `memory/window.py`, `memory/semantic.py`, `memory/context.py`, `tests/fakes.py` (`FakeEmbedding` only for now)
- Test: `tests/test_memory.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) Turn(task: str, answer: str)`.
- Produces: `SlidingWindow(max_turns: int)`:
  - `.add(task, answer)`: evicts the oldest turn when over `max_turns`.
  - `.turns() -> list[Turn]` returns a copy, oldest first.
  - `.clear()` and `__len__`.
- Produces: `class MemoryHit(BaseModel): id: str; text: str; distance: float`.
- Produces: `SemanticMemory(chroma_dir: Path, embedding_function=None, collection: str = "coder_runs")`:
  - `PersistentClient(path=str(chroma_dir))` with cosine space. Check the current chromadb way to set it: `configuration={"hnsw": {"space": "cosine"}}` in 1.x, otherwise `metadata={"hnsw:space": "cosine"}`.
  - A `None` embedding function means Chroma's default local one.
  - `.add(text: str, metadata: dict[str, str | int | float | bool]) -> str`: the id is `uuid4().hex`.
  - `.search(query: str, k: int, max_distance: float) -> list[MemoryHit]`: returns `[]` when the collection is empty; keeps only hits with `distance <= max_distance`, nearest first.
  - `.count() -> int`.
  - `.clear() -> None`: deletes and recreates the collection.
- Produces: `format_memory_context(turns: list[Turn], hits: list[MemoryHit]) -> str`:
  - Two markdown sections, `## Recent conversation` and `## Relevant memories from past runs`, each listing its items.
  - When there is nothing, it returns a line saying no prior context was found. It never returns an empty string, so the agent always sees the memory step happened.
- Produces, in `tests/fakes.py`: `FakeEmbedding`, a Chroma `EmbeddingFunction`. It is a deterministic bag-of-words hash into 64 dims, L2-normalised, so texts that share words are close. It needs no download.

- [ ] **Step 1: Write the failing tests**

```python
def test_window_evicts_oldest():
    w = SlidingWindow(max_turns=2)
    for i in range(3): w.add(f"t{i}", f"a{i}")
    assert [t.task for t in w.turns()] == ["t1", "t2"] and len(w) == 2

def test_semantic_roundtrip_and_threshold(tmp_path):
    m = SemanticMemory(tmp_path / ".chroma", embedding_function=FakeEmbedding())
    assert m.search("anything", k=3, max_distance=0.65) == []          # empty first run is fine
    m.add("Task: palindrome checker in palindrome.py", {"run_id": "r1", "files": "palindrome.py"})
    m.add("Task: fizzbuzz printer", {"run_id": "r2", "files": "fizz.py"})
    hits = m.search("make the palindrome checker ignore punctuation", k=3, max_distance=0.65)
    assert hits and "palindrome" in hits[0].text
    assert all("fizzbuzz" not in h.text for h in hits)

def test_semantic_persists_across_instances(tmp_path):
    SemanticMemory(tmp_path / ".chroma", embedding_function=FakeEmbedding()).add("Task: x", {"run_id": "r"})
    assert SemanticMemory(tmp_path / ".chroma", embedding_function=FakeEmbedding()).count() == 1

def test_clear(tmp_path):
    m = SemanticMemory(tmp_path / ".chroma", embedding_function=FakeEmbedding()); m.add("t", {"run_id": "r"})
    m.clear(); assert m.count() == 0

def test_context_formatting():
    ctx = format_memory_context([Turn("t1", "a1")], [MemoryHit(id="m1", text="Task: p", distance=0.2)])
    assert "## Recent conversation" in ctx and "t1" in ctx and "Task: p" in ctx
    assert format_memory_context([], []).strip() != ""
```

If `FakeEmbedding` puts fizzbuzz within 0.65 of the palindrome query, tune `FakeEmbedding`, not the threshold.

- [ ] **Step 2: Run it and see it fail.** `uv run pytest tests/test_memory.py -v` → FAIL.
- [ ] **Step 3: Implement the three modules and `FakeEmbedding`.**
- [ ] **Step 4: Run it and see it pass.** → 5 passed.
- [ ] **Step 5: Commit.** `git commit -m "feat(memory): sliding window, Chroma semantic store, context formatting"`

---

### Task 7: Result models and delivery verification

**Files:**
- Create: `agents/result.py`, `orchestration/state.py`, `orchestration/verify.py`
- Test: `tests/test_result.py`, `tests/test_verify.py`

**Interfaces:**
- Produces, in `agents/result.py`:
  - `Status = Literal["success", "failed", "iteration_limit", "error"]`.
  - `ExecutionResult(BaseModel)`: `executed: bool`, `target: str | None`, `exit_code: int | None`, `stdout: str = ""`, `stderr: str = ""`, `timed_out: bool = False`, `duration_s: float = 0.0`.
  - `CoderResult(BaseModel)`, with `model_config = ConfigDict(extra="forbid")`. Fields:
    - `code: str`
    - `explanation: str` (min_length=1)
    - `plan: list[str]` (min_length=1)
    - `result: ExecutionResult`
    - `status: Status`
    - `iterations: int` (0..10)
    - `files: dict[str, str]`
    - `memories_used: list[MemoryHit]`
    - `run_id: str`
    - `verified: bool`
    - `verification_notes: list[str] = []`
    - `error: str | None = None`
  - A `@model_validator(mode="after")`: if `status == "success"`, the run must satisfy `result.executed`, `exit_code == 0`, `not timed_out` and `verified`. Otherwise raise `ValueError("success requires a successful execution and verified delivery")`.
- Produces, in `orchestration/state.py`, `TypedDict`s:
  - `WriteInfo(seq: int, sha256: str)`
  - `ExecInfo(seq: int, target: str, ok: bool, exit_code: int | None, timed_out: bool, stdout: str, stderr: str, duration_s: float, loaded: dict[str, str])`
  - `ToolEvent(seq: int, iteration: int, name: str, args: dict, observation: str, ok: bool)`
  - `CoderState`:
    - `task: str`
    - `messages: Annotated[list[AnyMessage], add_messages]`
    - `iterations: int`
    - `seq: int`
    - `plan: list[str]`
    - `events: Annotated[list[ToolEvent], operator.add]`
    - `writes: Annotated[dict[str, WriteInfo], merge_dicts]`
    - `executions: Annotated[list[ExecInfo], operator.add]`
    - `memory_context: str`
    - `memories: Annotated[list[MemoryHit dicts], merge_hits]`, where `merge_hits` dedupes by `id`
    - `decision: dict | None`
    - `finish: dict | None`
    - `stop_reason: Literal["finished", "iteration_limit"] | None`
- Produces, in `orchestration/verify.py`: `@dataclass Verification(verified: bool, notes: list[str])`, and `verify_delivery(writes: dict[str, WriteInfo], executions: list[ExecInfo], workspace: Workspace, entry_point: str | None = None) -> Verification`.
  - Delivered files are the `.py` keys of `writes`.
  - A file is verified when some `ExecInfo` has `ok`, `seq > writes[f].seq`, and `loaded.get(f) == sha256_file(current file on disk)`.
  - Notes per failing file:
    - `"<f>: not executed"`: no ok run loaded it after its last write.
    - `"<f>: changed after last successful run"`: an ok run loaded it, but its hash now differs, or the run came before the last write.
    - `"<f>: missing on disk"`.
  - The entry point, if given, must be the `target` of a qualifying run; otherwise the note is `"<entry>: entry point was not run successfully"`.
  - `verified` = no notes. If no `.py` files were delivered, `verified` = (at least one ok execution exists).

- [ ] **Step 1: Write the failing tests**

`tests/test_result.py`:

```python
OK = ExecutionResult(executed=True, target="t.py", exit_code=0)
def make(**kw):
    base = dict(code="x", explanation="e", plan=["p"], result=OK, status="success", iterations=3,
                files={"t.py": "x"}, memories_used=[], run_id="r", verified=True)
    return CoderResult(**(base | kw))

def test_valid_success(): assert make().status == "success"
@pytest.mark.parametrize("kw", [
    {"verified": False},
    {"result": ExecutionResult(executed=False, target=None, exit_code=None)},
    {"result": ExecutionResult(executed=True, target="t.py", exit_code=1)},
    {"result": ExecutionResult(executed=True, target="t.py", exit_code=None, timed_out=True)},
])
def test_success_requires_real_verified_run(kw):
    with pytest.raises(ValidationError, match="success requires"): make(**kw)
def test_failed_may_be_unverified(): assert make(status="failed", verified=False).status == "failed"
def test_iterations_capped():
    with pytest.raises(ValidationError): make(iterations=11)
def test_required_fields_present():
    assert {"code", "explanation", "plan", "result"} <= set(CoderResult.model_fields)
```

`tests/test_verify.py`. A helper `w(ws, path, src, seq)` writes the file to disk and returns `{path: WriteInfo(seq=seq, sha256=...)}`. A helper `ex(seq, target, ok, loaded)` returns an `ExecInfo`.

```python
def test_write_a_run_write_b_is_unverified(tmp_path):          # spec §6.4 regression 1
    ws = Workspace(tmp_path / "ws")
    a = w(ws, "m.py", "print(1)", 1); run = ex(2, "m.py", True, {"m.py": a["m.py"]["sha256"]})
    b = w(ws, "m.py", "print(2)", 3)
    v = verify_delivery(b, [run], ws)
    assert not v.verified and v.notes == ["m.py: changed after last successful run"]

def test_unexecuted_broken_file_is_unverified(tmp_path):        # regression 2
    ws = Workspace(tmp_path / "ws")
    writes = w(ws, "palindrome.py", "def broken(:", 1) | w(ws, "ok.py", "print('ok')", 2)
    runs = [ex(3, "ok.py", True, {"ok.py": writes["ok.py"]["sha256"]})]
    v = verify_delivery(writes, runs, ws)
    assert not v.verified and "palindrome.py: not executed" in v.notes

def test_test_run_verifies_both_files(tmp_path):                 # regression 3
    ws = Workspace(tmp_path / "ws")
    writes = w(ws, "palindrome.py", "def p(s): return s==s[::-1]", 1) | w(ws, "test_palindrome.py", "...", 2)
    runs = [ex(3, "test_palindrome.py", True, {k: v["sha256"] for k, v in writes.items()})]
    assert verify_delivery(writes, runs, ws, entry_point="test_palindrome.py") == Verification(True, [])

def test_failed_run_does_not_verify(tmp_path):
    ws = Workspace(tmp_path / "ws"); writes = w(ws, "m.py", "x", 1)
    v = verify_delivery(writes, [ex(2, "m.py", False, {"m.py": writes["m.py"]["sha256"]})], ws)
    assert v.notes == ["m.py: not executed"]

def test_entry_point_must_be_run_target(tmp_path):
    ws = Workspace(tmp_path / "ws")
    writes = w(ws, "p.py", "x=1", 1) | w(ws, "main.py", "import p", 2)
    runs = [ex(3, "p.py", True, {"p.py": writes["p.py"]["sha256"]})]
    v = verify_delivery(writes, runs, ws, entry_point="main.py")
    assert "main.py: entry point was not run successfully" in v.notes
```

- [ ] **Step 2: Run it and see it fail.** `uv run pytest tests/test_result.py tests/test_verify.py -v` → FAIL.
- [ ] **Step 3: Implement `agents/result.py`, `orchestration/state.py`, `orchestration/verify.py`.**
- [ ] **Step 4: Run it and see it pass.** → 13 passed.
- [ ] **Step 5: Commit.** `git commit -m "feat: CoderResult validation and delivery verification"`

---

### Task 8: The graph: routing, nodes, finalize

**Files:**
- Create: `orchestration/routing.py`, `orchestration/finalize.py`, `orchestration/graph.py`, `agents/events.py`, `agents/prompts.py`; extend `tests/fakes.py`
- Test: `tests/test_routing.py`, `tests/test_graph.py`

**Interfaces:**
- Consumes: Tasks 2–7.
- Produces, in `agents/events.py`: `StepEvent(BaseModel)` with:
  - `kind: Literal["start","memory","thought","plan","action","observation","nudge","limit","result","error"]`
  - `iteration: int`
  - `max_iterations: int`
  - `data: dict = {}`
- Produces, in `orchestration/routing.py`:
  - `@dataclass Decision`:
    - `kind: Literal["finish","execute","nudge","limit"]`
    - `run_calls: list[dict]`: ordinary tool calls to execute, in emitted order
    - `rejections: list[tuple[str, str]]`: (tool_call_id, observation) for calls answered without running
    - `finish_args: dict | None`
  - `classify(message: AIMessage, iterations: int, max_iterations: int) -> Decision`. It implements spec §6.1's precedence (*n* = `iterations`, already incremented for this call):
    1. A lone `finish` with valid `FinishArgs` → `finish`, at any *n*. With invalid args, it becomes a rejection `"finish rejected: <validation error>"` and is treated as case 4.
    2. `finish` alongside other calls → reject finish with `"finish must be called alone, after your last action"`, then apply case 3 to the rest.
    3. Ordinary calls → `execute` if *n* < max, else `limit`.
    4. No tool calls, or only a rejected finish → `nudge` if *n* < max, else `limit`.
  - `extract_plan(text: str) -> list[str]`: lines matching `^\s*\d+[.)]\s+(.+)` → the captured text.
- Produces, in `orchestration/finalize.py`: `build_result(state: CoderState, *, run_id: str, workspace: Workspace) -> CoderResult`.
  - `files`: every key in `state["writes"]` that still exists → its current disk content.
  - `code`: with one file, its content; with several, each prefixed `# ===== <path> =====`, joined with a blank line; with none, `""`.
  - `result`: built from `executions[-1]`, or `ExecutionResult(executed=False, target=None, exit_code=None)`.
  - `verified` and `notes`: from `verify_delivery`, using `finish.entry_point`.
  - Explanation and plan: from `finish` args when finished. At the limit, a template: `"Stopped after {n} of {max} iterations without calling finish. Last action: {name}. Last observation: {first 300 chars}."` The plan becomes `state["plan"] or ["No plan was recorded before the run stopped."]`.
  - Status:
    - `iteration_limit` if `stop_reason == "iteration_limit"`.
    - Otherwise `success` if `result.ok and verified`.
    - Otherwise `failed`; append the reason to `verification_notes`, including `"last execution failed"`.
  - `memories_used` comes from `state["memories"]`.
- Produces, in `orchestration/graph.py`:
  - `@dataclass GraphDeps`:
    - `model: BaseChatModel` (already `bind_tools(TOOL_SPECS)`)
    - `tool_ctx: ToolContext`
    - `semantic: SemanticMemory`
    - `window: SlidingWindow`
    - `max_iterations: int`
    - `top_k: int`
    - `max_distance: float`
    - `run_id: str`
  - `initial_state(task: str) -> CoderState`
  - `build_graph(deps: GraphDeps) -> CompiledStateGraph`, with nodes `retrieve_memory`, `agent`, `tools` and `finalize`. Edges:
    - `START→retrieve_memory→agent`
    - `agent →(finish|limit)→ finalize`, else `→ tools → retrieve_memory`
    - `finalize→END`
  - Invoke with `config={"recursion_limit": 3 * max_iterations + 5}`.
- Produces, in `tests/fakes.py`:
  - `tool_call(name, **args) -> dict`: a LangChain tool_call dict with a unique `id`.
  - `ai(text="", *calls) -> AIMessage`.
  - `ScriptedChatModel(BaseChatModel)`:
    - built from `responses: list[AIMessage | Exception]`
    - `bind_tools(...)` returns self
    - each `_generate` pops the next response (raising it if it's an Exception) and appends the messages it received to `self.calls`
    - running out of responses raises `AssertionError("model called more times than scripted")`.
  - Each returned `AIMessage` needs a unique `id`. `add_messages` replaces messages that share an id, so scripts must build a new message per response (`[ai(...) for _ in range(n)]`, never `[ai(...)] * n`). `_generate` should also deep-copy the message it returns.

**Node behaviour:**
- **retrieve_memory:**
  - The query is the task plus the last `ToolMessage`/nudge content, cut to 500 characters.
  - Calls `semantic.search`, then `format_memory_context(window.turns(), hits)`.
  - Returns `memory_context` and `memories`, and emits StepEvent `memory` with `{"hits": [...], "window_turns": n}`.
- **agent:**
  - Sends `[SystemMessage(SYSTEM_PROMPT + "\n\n" + memory_context)] + state["messages"]`. Memory reaches **every** call this way, and the system message is never stored in `messages`.
  - Increments `iterations` and emits `thought` (`message.text`).
  - Updates `plan` when `extract_plan` is non-empty, and emits `plan`.
  - Stores `decision`. On `limit`, it also sets `stop_reason="iteration_limit"` and emits `limit`.
- **tools:**
  1. Appends a `ToolMessage` for each rejection.
  2. Runs `run_calls` sequentially. Each call emits `action`, increments `seq`, runs `run_action`, records the `ToolEvent` / `WriteInfo` / `ExecInfo`, appends a `ToolMessage`, and emits `observation`.
  3. On a `ToolError`, the observation is `"error: <msg>"` (`ok=False`), and every remaining call gets `"skipped: an earlier call in this batch failed"`.
  4. A `nudge` decision appends `HumanMessage("Observation: no tool was called. Call a tool to make progress, or call finish (alone) when the deliverable has been executed successfully.")` after any rejection ToolMessages, and emits `nudge`.
- **finalize:** on `finish`, sets `stop_reason="finished"`. Stores `build_result(...)`'s dump in state under `result` (add `result: dict | None` to `CoderState`), and emits `result`.
- **Events:** emitted with `get_stream_writer()(event.model_dump())`.

**`agents/prompts.py`:** the `SYSTEM_PROMPT` content (wording is your call):
- Coder agent; ReAct.
- Start the first turn with a numbered plan.
- Each turn: short reasoning, then tool calls.
- Paths are relative to the workspace with no `workspace/` prefix.
- Prefer writing tests (`test_*.py`) and running them with `exec_python`.
- Only call `finish` alone, after the final files have been executed successfully in their final form; pass `entry_point`.
- Up to 10 model turns.
- Use the memory section to build on past runs (read the files it names).

- [ ] **Step 1: Write the failing routing tests** `tests/test_routing.py`

```python
W = tool_call("write_file", path="a.py", content="x"); F = tool_call("finish", explanation="e", plan=["p"])
def test_finish_alone(): d = classify(ai("", F), 3, 10); assert d.kind == "finish" and d.finish_args["plan"] == ["p"]
def test_finish_alone_at_limit_is_honoured(): assert classify(ai("", F), 10, 10).kind == "finish"
def test_finish_mixed_is_rejected_others_run():
    d = classify(ai("", W, F), 3, 10)
    assert d.kind == "execute" and [c["name"] for c in d.run_calls] == ["write_file"]
    assert d.rejections == [(F["id"], "finish must be called alone, after your last action")]
def test_tools_at_limit_not_executed():
    d = classify(ai("", W), 10, 10); assert d.kind == "limit" and d.run_calls == []
def test_text_only_nudges_then_limits():
    assert classify(ai("thinking"), 4, 10).kind == "nudge"
    assert classify(ai("thinking"), 10, 10).kind == "limit"
def test_invalid_finish_args_is_case_4():
    bad = tool_call("finish", explanation="e")            # missing plan
    d = classify(ai("", bad), 2, 10); assert d.kind == "nudge" and d.rejections[0][1].startswith("finish rejected:")
def test_extract_plan(): assert extract_plan("Plan:\n1. write\n2) test\nok") == ["write", "test"]
```

- [ ] **Step 2: Run it and see it fail.** `uv run pytest tests/test_routing.py -v` → FAIL.
- [ ] **Step 3: Implement `orchestration/routing.py`.**
- [ ] **Step 4: Run it and see it pass.** → 7 passed.

- [ ] **Step 5: Write the failing graph tests** `tests/test_graph.py`

The fixture `deps(settings, model)` builds a `GraphDeps` with:
- a `Workspace` on `settings.workspace_dir`
- `record_dir = settings.state_dir / "exec"`
- `SemanticMemory(..., FakeEmbedding())`
- `SlidingWindow(10)`
- `max_iterations=settings.max_iterations`

The helper `run_graph(deps) -> (final_state, events)` streams with `stream_mode=["custom", "values"], version="v2"` and collects custom data and the last values chunk. Check the chunk shape against the LangGraph streaming docs.

```python
PAL = "def is_pal(s):\n    return s == s[::-1]\n"
TEST = "from palindrome import is_pal\ndef test_it():\n    assert is_pal('aba')\n"

def test_happy_path_success(settings):
    m = ScriptedChatModel([
        ai("1. write module\n2. write test\n3. run test",
           tool_call("write_file", path="palindrome.py", content=PAL),
           tool_call("write_file", path="test_palindrome.py", content=TEST),
           tool_call("exec_python", path="test_palindrome.py")),       # same batch: runs the fresh files
        ai("done", tool_call("finish", explanation="Checks palindromes", plan=["write", "test"],
                             entry_point="test_palindrome.py")),
    ])
    state, events = run_graph(deps(settings, m))
    r = CoderResult.model_validate(state["result"])
    assert r.status == "success" and r.verified and r.iterations == 2
    assert r.files["palindrome.py"] == (settings.workspace_dir / "palindrome.py").read_text()
    assert r.result.exit_code == 0 and "1 passed" in r.result.stdout

def test_memory_retrieved_before_every_call(settings):
    d = deps(settings, ScriptedChatModel([ai("thinking"), ai("still"), ai("", tool_call("exec_python", code="print(1)")),
                                          ai("", tool_call("finish", explanation="e", plan=["p"]))]))
    d.semantic.add("Task: earlier palindrome work", {"run_id": "old"})
    d.window.add("earlier task", "earlier answer")
    state, events = run_graph(d)
    assert [e["kind"] for e in events].count("memory") == 4 == len(d.model.calls)
    for call in d.model.calls:
        assert "earlier answer" in call[0].content and "## Recent conversation" in call[0].content

def test_iteration_limit_forced(settings):
    s = settings.model_copy(update={"max_iterations": 3})
    m = ScriptedChatModel([ai("", tool_call("exec_python", code="print(1)")) for _ in range(3)])
    state, events = run_graph(deps(s, m))
    r = CoderResult.model_validate(state["result"])
    assert r.status == "iteration_limit" and r.iterations == 3 and len(m.calls) == 3
    assert len([e for e in events if e["kind"] == "action"]) == 2     # 3rd call's tools not executed
    assert r.explanation.startswith("Stopped after 3 of 3 iterations")

def test_default_limit_is_ten(settings):
    m = ScriptedChatModel([ai("", tool_call("read_file", path="nope.py")) for _ in range(10)])
    state, _ = run_graph(deps(settings, m))
    assert state["result"]["status"] == "iteration_limit" and len(m.calls) == 10

def test_error_mid_batch_skips_rest(settings):
    m = ScriptedChatModel([
        ai("", tool_call("read_file", path="../escape"), tool_call("write_file", path="a.py", content="x")),
        ai("", tool_call("finish", explanation="e", plan=["p"])),
    ])
    state, _ = run_graph(deps(settings, m))
    obs = [msg.content for msg in state["messages"] if isinstance(msg, ToolMessage)]
    assert obs[0].startswith("error:") and obs[1] == "skipped: an earlier call in this batch failed"
    assert not (settings.workspace_dir / "a.py").exists()

def test_write_a_exec_write_b_finish_is_failed(settings):          # spec §6.4 regression via the graph
    m = ScriptedChatModel([
        ai("", tool_call("write_file", path="m.py", content="print(1)"), tool_call("exec_python", path="m.py")),
        ai("", tool_call("write_file", path="m.py", content="print(2)")),
        ai("", tool_call("finish", explanation="e", plan=["p"])),
    ])
    r = CoderResult.model_validate(run_graph(deps(settings, m))[0]["result"])
    assert r.status == "failed" and not r.verified and "m.py: changed after last successful run" in r.verification_notes

def test_unexecuted_broken_file_is_failed(settings):              # spec §6.4 regression 2 via the graph
    m = ScriptedChatModel([
        ai("", tool_call("write_file", path="palindrome.py", content="def broken(:"),
               tool_call("write_file", path="ok.py", content="print('ok')"),
               tool_call("exec_python", path="ok.py")),
        ai("", tool_call("finish", explanation="e", plan=["p"])),
    ])
    r = CoderResult.model_validate(run_graph(deps(settings, m))[0]["result"])
    assert r.status == "failed" and "palindrome.py: not executed" in r.verification_notes
```

- [ ] **Step 6: Run it and see it fail.** `uv run pytest tests/test_graph.py -v` → FAIL.
- [ ] **Step 7: Implement `agents/events.py`, `agents/prompts.py`, `orchestration/finalize.py`, `orchestration/graph.py`.**
- [ ] **Step 8: Run it and see it pass.** `uv run pytest tests/test_routing.py tests/test_graph.py -v` → 14 passed.
- [ ] **Step 9: Commit.** `git commit -m "feat(orchestration): bounded ReAct graph with memory-first calls and validated finalize"`

---

### Task 9: `CoderAgent`: lock, reset generation, memory write-back, errors

**Files:**
- Create: `agents/model.py`, `agents/coder_agent.py`
- Test: `tests/test_coder_agent.py`, `tests/test_model.py`

**Interfaces:**
- Consumes: Tasks 1–8.
- Produces, in `agents/model.py`:
  - `make_model(settings: Settings) -> BaseChatModel`: `init_chat_model(settings.model, max_tokens=settings.max_tokens)`. Pass no temperature.
  - `provider_prerequisite(model: str) -> str | None`. Returns `None` when usable, else a reason:
    - `anthropic:` needs `ANTHROPIC_API_KEY`.
    - `openai:` needs `OPENAI_API_KEY`.
    - `ollama:` needs `GET {OLLAMA_BASE_URL or http://localhost:11434}/api/tags` to answer within 2 s and list the model name.
    - Any other prefix → `"unsupported provider prefix"`.
  - The reason names the missing prerequisite, e.g. `"ANTHROPIC_API_KEY is not set (needed for anthropic:claude-sonnet-5-5)"`.
- Produces, in `agents/coder_agent.py`:
  - `CoderAgent(settings: Settings | None = None, *, model: BaseChatModel | None = None, embedding_function=None)`. When `model` is None, call `make_model(settings)` lazily on the first run. The agent owns `workspace`, `window: SlidingWindow(settings.window_turns)`, `semantic`, and `generation_seen: int`.
  - `CoderAgent.stream(task: str) -> Iterator[StepEvent]`:
    1. Enter `RunLock(settings.state_dir)`. A busy lock raises `WorkspaceBusyError` on the first `next()`, before anything is yielded.
    2. Read `ResetGeneration`. If it differs from `generation_seen`, clear `window` and adopt the new value.
    3. Yield `start` with `{"task", "run_id", "model"}`.
    4. Run the graph, yielding each custom event as a `StepEvent`.
    5. On any exception from the model or graph, yield `error` and then `result` with an `error`-status `CoderResult`:
       - `explanation = f"Run failed: {type(e).__name__}: {e}"`
       - `plan` = the state's plan, or `["Run failed before a plan was recorded."]`
       - `result` = not executed
       - `verified=False`
       - `error=str(e)`
    6. After a non-error result: `semantic.add(memory_document(task, result), {"run_id", "status", "files": ",".join(files), "created_at": iso})` and `window.add(task, result.explanation)`.
    7. The lock is released in `finally`.
  - `memory_document(task, result) -> str`, which contains:
    - `Task: …`
    - `Outcome: <status>`
    - `Files: …`
    - `Plan: …`
    - `Summary: <explanation, first 500 characters>`
    - `Defines: <top-level def/class names of the delivered .py files, via ast; parse failures skipped>`
  - `CoderAgent.run(task: str) -> CoderResult`: drains `stream` and returns the last `result` event's `CoderResult`.
  - `CoderAgent.reset_memory() -> None`: under `RunLock`, `semantic.clear()`, `workspace.clear()`, `ResetGeneration.bump()`, `window.clear()`, and set `generation_seen` to the new value. Busy → `WorkspaceBusyError`.
  - `run_coder(task: str, settings: Settings | None = None) -> CoderResult`.

- [ ] **Step 1: Write the failing tests**

First, add two shared helpers to `tests/fakes.py`; Tasks 10 and 12 reuse them:
- `happy() -> list[AIMessage]`: returns a **fresh** copy of the two-response palindrome script from `test_happy_path_success` in Task 8. Move `PAL` and `TEST` into `tests/fakes.py` too.
- `agent(settings, *responses) -> CoderAgent`: builds `CoderAgent(settings, model=ScriptedChatModel(list(responses)), embedding_function=FakeEmbedding())`.

`CoderAgent` exposes the model it uses as `.model`, so tests can read `ScriptedChatModel.calls`.

Then the tests, in `tests/test_coder_agent.py`:

```python
def test_run_returns_result_and_writes_memory(settings):
    a = agent(settings, *happy())
    r = a.run("Write a palindrome checker with tests")
    assert r.status == "success" and a.semantic.count() == 1 and len(a.window) == 1

def test_second_task_sees_first_in_memory(settings):
    a = agent(settings, *happy(), ai("", tool_call("finish", explanation="e", plan=["p"])))
    a.run("Write a palindrome checker with tests")
    r = a.run("Now make the palindrome checker ignore punctuation")
    assert any("palindrome" in h.text for h in r.memories_used)
    assert "Write a palindrome checker" in a.model.calls[-1][0].content

def test_busy_workspace_raises_before_any_event(settings):
    with RunLock(settings.state_dir):
        # a fresh FileLock instance on the same path is a separate open file description, so it conflicts
        with pytest.raises(WorkspaceBusyError):
            next(agent(settings).stream("x"))

def test_reset_refused_while_run_holds_lock(settings):
    with RunLock(settings.state_dir):
        with pytest.raises(WorkspaceBusyError):
            agent(settings).reset_memory()

def test_model_error_becomes_error_result_and_releases_lock(settings):      # Review Focus 5
    a = agent(settings, RuntimeError("402 insufficient credits"))
    r = a.run("anything")
    assert r.status == "error" and "402 insufficient credits" in r.error and not r.verified
    with RunLock(settings.state_dir):
        pass

def test_reset_from_another_process_clears_window(settings):
    a = agent(settings, *happy(), ai("", tool_call("finish", explanation="e", plan=["p"])))
    a.run("Write a palindrome checker with tests"); assert len(a.window) == 1
    subprocess.run([sys.executable, "-c", RESET_SNIPPET, str(settings.project_dir)], check=True, cwd=REPO_ROOT)
    a.run("next task")
    assert "Write a palindrome checker" not in a.model.calls[-1][0].content
    assert a.semantic.count() == 1            # only the run after the reset
```

`RESET_SNIPPET` builds a `CoderAgent(Settings.from_env(project_dir=argv[1]), model=object(), embedding_function=FakeEmbedding())` and calls `reset_memory()`. `model=object()` avoids `make_model`.

`tests/test_model.py`:

```python
def test_anthropic_prereq(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert "ANTHROPIC_API_KEY" in provider_prerequisite("anthropic:claude-sonnet-5-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x"); assert provider_prerequisite("anthropic:claude-sonnet-5-5") is None
def test_unknown_prefix(): assert "unsupported" in provider_prerequisite("foo:bar")
```

- [ ] **Step 2: Run it and see it fail.** `uv run pytest tests/test_coder_agent.py tests/test_model.py -v` → FAIL.

- [ ] **Step 3: Implement `agents/model.py` and `agents/coder_agent.py`.**

LangSmith: if `LANGSMITH_TRACING=true`, LangChain picks it up from the env, so no code is needed. Document it in Task 13.

The same-process busy tests (`RunLock` held in the test while the agent tries to lock) depend on `filelock` not sharing one lock between two instances on the same path. If the installed filelock shares them, hold the lock from a subprocess using Task 2's `HOLD_LOCK_SNIPPET` instead. Do not change `RunLock` to pass the test.

- [ ] **Step 4: Run it and see it pass.** → 8 passed. Then `uv run pytest` → all unit tests pass.
- [ ] **Step 5: Commit.** `git commit -m "feat(agents): CoderAgent with run lock, reset generation, memory write-back"`

---

### Task 10: CLI

**Files:**
- Create: `app/render.py`, `app/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `CoderAgent`, `StepEvent`, `CoderResult`, `Settings`, `provider_prerequisite`, `WorkspaceBusyError`.
- Produces:
  - `render_event(console: Console, event: StepEvent) -> None`, one line or panel per kind, prefixed `[n/max]`:
    - memory: hit count + top hit's first line
    - thought: dim text
    - plan: numbered
    - action: tool + args, with content shortened to 80 characters
    - observation: first 15 lines
    - nudge / limit / error: highlighted
  - `render_result(console: Console, result: CoderResult) -> None`: a rich panel with the status badge, verified ✓/✗ with notes, plan, explanation, files, the last execution (exit code, timed_out, stdout tail), the code (syntax-highlighted, first 60 lines), and memories used.
  - `main(argv: list[str] | None = None, *, agent: CoderAgent | None = None) -> int`.
- Arguments:
  - positional `task` (optional when `--demo` or `--reset-memory` is given alone)
  - `--model`, `--max-iterations INT`, `--timeout FLOAT`
  - `--reset-memory`
  - `--json` (prints only `CoderResult.model_dump_json(indent=2)` to stdout; events go to stderr)
  - `--demo`
- Exit codes:
  - 0: every result is success
  - 1: any result is failed / iteration_limit / error
  - 2: a usage error, a settings `ValidationError` (print its message, e.g. "between 1 and 10"), `WorkspaceBusyError`, or a missing provider prerequisite (checked before the run, only when `agent` is not injected)
- `--demo` runs the two tasks below in one process, with a rule between them titled "Follow-up: memory recall". These strings are exact and also reused in Task 11:

  `DEMO_TASK_1 = "Create palindrome.py with a function is_palindrome(text: str) -> bool that ignores case and spaces, plus test_palindrome.py with pytest tests covering at least four cases. Run the tests."`

  `DEMO_TASK_2 = "Now make the palindrome checker also ignore punctuation, add tests for that, and run all the tests."`

- [ ] **Step 1: Write the failing tests**

```python
def test_json_output_is_valid_result(settings, capsys):
    code = main(["--json", "Write a palindrome checker"], agent=agent(settings, *happy()))
    out = capsys.readouterr().out
    assert code == 0 and CoderResult.model_validate_json(out).status == "success"

def test_max_iterations_above_cap_rejected(capsys):
    assert main(["--max-iterations", "11", "task"]) == 2
    assert "between 1 and 10" in capsys.readouterr().err

def test_limit_exits_1(settings):
    s = settings.model_copy(update={"max_iterations": 2})
    a = CoderAgent(s, model=ScriptedChatModel([ai("thinking") for _ in range(2)]), embedding_function=FakeEmbedding())
    assert main(["task"], agent=a) == 1

def test_busy_exits_2(settings, capsys):
    with RunLock(settings.state_dir):
        assert main(["task"], agent=agent(settings)) == 2
    assert "workspace busy" in capsys.readouterr().err

def test_model_error_exits_1_without_traceback(settings, capsys):      # Review Focus 5
    assert main(["task"], agent=agent(settings, RuntimeError("401 invalid x-api-key"))) == 1
    err = capsys.readouterr()
    assert "Traceback" not in err.out + err.err and "401 invalid x-api-key" in err.out + err.err

def test_missing_key_exits_2(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False); monkeypatch.setenv("CODER_PROJECT_DIR", str(tmp_path))
    assert main(["task"]) == 2 and "ANTHROPIC_API_KEY" in capsys.readouterr().err
```

- [ ] **Step 2: Run it and see it fail.** `uv run pytest tests/test_cli.py -v` → FAIL.
- [ ] **Step 3: Implement `app/render.py` and `app/cli.py`.**
- [ ] **Step 4: Run it and see it pass.** → 6 passed. Then run `uv run coder --help` and check that every flag is listed.
- [ ] **Step 5: Commit.** `git commit -m "feat(app): rich CLI with demo, json output and exit codes"`

---

### Task 11: Live smoke test and `make verify`

**Files:**
- Create: `tests/live/__init__.py`, `tests/live/test_smoke.py`; modify `tests/conftest.py`
- Test: the files above

**Interfaces:**
- Consumes: `CoderAgent`, `provider_prerequisite`, and `DEMO_TASK_1` / `DEMO_TASK_2` from `app.cli`.
- Produces: a fixture `live_settings(tmp_path) -> Settings`:
  - calls `Settings.from_env(project_dir=tmp_path)` and `provider_prerequisite(settings.model)`
  - if a reason is returned: when `os.environ.get("REQUIRE_LIVE") == "1"`, call `pytest.fail(f"live test required but {reason}")`; otherwise `pytest.skip(reason)`
- These tests use the real default embedding function. Chroma downloads its small ONNX model on first use; that's expected.

- [ ] **Step 1: Write the live tests** (marked `@pytest.mark.live`)

```python
def test_smoke_structured_output(live_settings):
    r = CoderAgent(live_settings).run(DEMO_TASK_1)
    assert r.status == "success", r.verification_notes
    assert r.verified and r.result.executed and r.result.exit_code == 0 and not r.result.timed_out
    assert r.code.strip() and r.explanation.strip() and r.plan and all(p.strip() for p in r.plan)
    assert 1 <= r.iterations <= 10
    for path, content in r.files.items():
        assert (live_settings.workspace_dir / path).read_text() == content
        assert content in r.code
    assert "palindrome.py" in r.files and any(p.startswith("test_") for p in r.files)

def test_memory_follow_up(live_settings):
    CoderAgent(live_settings).run(DEMO_TASK_1)
    r = CoderAgent(live_settings).run(DEMO_TASK_2)     # fresh instance: proves Chroma cross-session recall
    assert r.status == "success", r.verification_notes
    assert any("palindrome" in h.text.lower() for h in r.memories_used)
```

- [ ] **Step 2: Run without a key to check the gating.**

Run `env -u ANTHROPIC_API_KEY uv run pytest tests/live -rs`. Expected: 2 skipped, and the reason names `ANTHROPIC_API_KEY`.

Then run `env -u ANTHROPIC_API_KEY REQUIRE_LIVE=1 uv run pytest tests/live -rs`. Expected: 2 failed with "live test required but ANTHROPIC_API_KEY is not set".

- [ ] **Step 3: Run live.** `op run --env-file=.env.op -- make verify` → all tests pass and the `-rs` summary shows 0 skipped.

If the model isn't reliably passing `test_smoke_structured_output`, iterate on `SYSTEM_PROMPT`, not on the assertions. Record the number of attempts in the commit message.

- [ ] **Step 4: Commit.** `git commit -m "test: live smoke and memory follow-up, REQUIRE_LIVE gating"`

---

### Task 12: Web UI (layout B, read-only)

**Files:**
- Create: `app/web.py`, `web/index.html`, `web/app.js`, `web/style.css`
- Test: `tests/test_web.py`

**Interfaces:**
- Consumes: `CoderAgent`, `StepEvent`, `Workspace`, `WorkspaceError`, `WorkspaceBusyError`.
- Produces: `create_app(agent: CoderAgent) -> FastAPI` and `main() -> None`.
  - `main` runs uvicorn on `CODER_WEB_HOST` (default `127.0.0.1`) and `CODER_WEB_PORT` (default `8000`) with a lazily created default agent.
- Endpoints:
  - `GET /` → `web/index.html`, plus static files under `/static`.
  - `GET /api/files` → `{"files": [{"path", "size", "touched"}]}`. `touched` means written in the current or last run, per the agent's last `writes`. Expose that on `CoderAgent` as `last_written: set[str]`, updated on `action` events for `write_file`.
  - `GET /api/file?path=…` → `{"path", "content"}`. Errors: 400 on `WorkspaceError`, 404 when missing.
  - `GET /api/status` → `{"running": bool}`.
  - `POST /api/run` with body `{"task": str}`:
    - Create `agent.stream(task)` and pull the first event in a threadpool. `WorkspaceBusyError` → **409** `{"detail": "workspace busy"}`.
    - Otherwise return a `StreamingResponse(media_type="text/event-stream")` of `data: <StepEvent json>\n\n` frames. The first frame is the event already pulled; the rest come via `starlette.concurrency.iterate_in_threadpool`.
- **Frontend** (plain HTML/JS, no build step; highlight.js 11.9.0 from cdnjs for Python):
  - Dark theme, layout B (spec §4).
  - Top bar: task input + Run. Run is disabled while running or while `/api/status` reports running.
  - Left: file tree from `/api/files`, refreshed after each `observation` event; touched files highlighted.
  - Centre: read-only code viewer for the selected file, auto-selecting the most recently written file, with a terminal strip below it showing the latest `exec_python` observation.
  - Right: an event timeline (memory / thought / plan / action / observation / nudge / limit) and an `n/10` counter.
  - On `result`: a result card with status, verified, explanation, plan, and memories used.
  - Read the POST stream with `fetch` + `ReadableStream` (EventSource cannot POST).

- [ ] **Step 1: Write the failing tests**

```python
def client(settings, *responses):
    return TestClient(create_app(agent(settings, *responses)))

def test_run_streams_events_and_result(settings):
    with client(settings, *happy()).stream("POST", "/api/run", json={"task": "palindrome"}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "start" and "memory" in kinds and kinds[-1] == "result"
    assert events[-1]["data"]["status"] == "success"

def test_busy_is_409(settings):
    c = client(settings)
    with RunLock(settings.state_dir):
        assert c.post("/api/run", json={"task": "x"}).status_code == 409

def test_file_endpoints(settings):
    c = client(settings, *happy()); c.post("/api/run", json={"task": "palindrome"})
    files = {f["path"]: f for f in c.get("/api/files").json()["files"]}
    assert files["palindrome.py"]["touched"]
    assert "def is_pal" in c.get("/api/file", params={"path": "palindrome.py"}).json()["content"]
    assert c.get("/api/file", params={"path": "../pyproject.toml"}).status_code == 400
    assert c.get("/api/file", params={"path": "nope.py"}).status_code == 404

def test_index_served(settings):
    assert "Coder" in client(settings).get("/").text
```

- [ ] **Step 2: Run it and see it fail.** `uv run pytest tests/test_web.py -v` → FAIL.
- [ ] **Step 3: Implement `app/web.py` and the three frontend files.**
- [ ] **Step 4: Run it and see it pass.** → 4 passed.

- [ ] **Step 5: Check it in a real browser**

1. Start `op run --env-file=.env.op -- uv run coder-web`.
2. Open `http://127.0.0.1:8000` (use the `run` or `playwright-cli` skill) and run `DEMO_TASK_1`.
3. Confirm all of these:
   - files appear in the tree as they're written
   - the code viewer shows `palindrome.py`
   - the terminal strip shows pytest output
   - the counter advances
   - the result card shows success and verified ✓
4. Take a screenshot and save it to `docs/images/web-ui.png`.

- [ ] **Step 6: Commit.** `git commit -m "feat(web): read-only mini-IDE with live SSE agent stream"`

---

### Task 13: Docker, docs, README, evidence

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `docs/architecture.md`, `docs/usage.md`, `docs/evidence/` (`cli-transcript.txt`, `coder-result.json`, `verify-output.txt`); rewrite `README.md`
- Test: the commands below

**Interfaces:**
- Consumes: everything.

- [ ] **Step 1: Write the Dockerfile**

1. Base image: `python:3.12-slim`.
2. `COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /bin/`.
3. Copy `pyproject.toml` and `uv.lock`, then `uv sync --frozen --no-dev`.
4. Copy the source.
5. Pre-download Chroma's default embedding model with `uv run python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warmup'])"`.
6. `ENV CODER_WEB_HOST=0.0.0.0 PATH=/app/.venv/bin:$PATH`, `EXPOSE 8000`, `CMD ["coder-web"]`.

- [ ] **Step 2: Write `docker-compose.yml`**

Service `web`:
- ports `"127.0.0.1:8000:8000"`
- environment: `ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY}` and `CODER_MODEL: ${CODER_MODEL:-anthropic:claude-sonnet-5-5}`
- volumes `./workspace:/app/workspace`, `./.chroma:/app/.chroma`, `./.coder:/app/.coder`

Service `coder`: same image and volumes, `entrypoint: ["coder"]`, `profiles: ["cli"]`.

Verify:
- `docker compose build`
- `docker compose run --rm coder --help` (flags print)
- `op run --env-file=.env.op -- docker compose run --rm coder "print hello world from hello.py and run it"` (status success)

If Docker isn't installed on this Mac, say so in the task report and leave these steps unchecked. Don't skip silently.

- [ ] **Step 3: Write `docs/architecture.md` and `docs/usage.md`**

`architecture.md`: a Mermaid diagram of the graph (matching spec §6.1), then sections on tools, the execution record and verification, memory, routing precedence, and the Module 2–5 extension points.

`usage.md`: CLI flags, the web UI, the demo, `--reset-memory`, the probes, and how to switch to `openai:` or `ollama:`.

- [ ] **Step 4: Write `README.md`** (spec §6.8)

Sections in order:
1. Pitch + screenshot.
2. Quickstart: the 1Password path (`op run --env-file=.env.op -- make demo`), the plain `.env` path (`cp .env.example .env`), and the pip path (`pip install -r requirements.txt && pip install -e . --no-deps`).
3. CLI usage.
4. Web UI.
5. Architecture (embed the Mermaid diagram).
6. Implementation decisions (LangGraph hand-built graph, Claude Sonnet via `init_chat_model`, Chroma default local embeddings, execution-record verification, the shared workspace + lock).
7. **Rubric evidence**, a table with columns *criterion · where in code · how to observe*:
   - C1.1: tree + `make setup && uv run coder --help`
   - C1.2: `orchestration/graph.py` retrieve_memory, `make demo` second task
   - C1.3: the timeout probe `uv run coder --timeout 2 "write loop.py containing while True: pass and run it"` → observation `timed_out=True`; `tests/test_exec_python.py`
   - C1.4: the probe `uv run coder --max-iterations 3 "<multi-step task>"` → status `iteration_limit`, at most 3 model calls; `tests/test_graph.py::test_iteration_limit_forced`
   - C1.5: `agents/result.py` validator + `orchestration/verify.py`; `uv run coder --json …`
   - C1.6: `tests/live/test_smoke.py`, `make verify`
8. Known limitations: subprocess is not a security sandbox; executed code has host filesystem and network access; one run per workspace.
9. Roadmap for Modules 2–5.
10. LangSmith (optional).

- [ ] **Step 5: Capture evidence at the final code state**

1. `op run --env-file=.env.op -- uv run coder --reset-memory`
2. `op run --env-file=.env.op -- uv run coder --demo 2>&1 | tee docs/evidence/cli-transcript.txt`
3. `op run --env-file=.env.op -- uv run coder --json "<DEMO_TASK_1>" > docs/evidence/coder-result.json`
4. `op run --env-file=.env.op -- make verify 2>&1 | tee docs/evidence/verify-output.txt`

Check the transcript shows memory recall on the second task, and that `verify-output.txt` shows 0 skipped.

- [ ] **Step 6: Final checks**

1. `make requirements`, then `git diff --exit-code requirements.txt`: requirements are in sync.
2. In a fresh clone (`git clone . /private/tmp/claude-…/scratchpad/clone`), run `uv sync && uv run pytest -q`. The unit tests pass and the live tests skip.
3. Check `git ls-files` contains no `.env`, `workspace/`, `.chroma/` or `.coder/`.

- [ ] **Step 7: Commit.** `git commit -m "docs: README with rubric evidence, architecture, Docker, captured evidence"`

---

### Task 14: Publish (requires Brandon's go-ahead)

- [ ] **Step 1: Ask Brandon to confirm** creating the **public** repo `darwinz/multi-agent-dev-team` and pushing. Also remind him of the open questions for course staff: whether to submit the lab forks or this repo, and the "1 April 2026" due date.
- [ ] **Step 2: On a yes:** `gh repo create darwinz/multi-agent-dev-team --public --source . --push`.
- [ ] **Step 3: Report** the repo URL and `git rev-parse HEAD`. That's the commit SHA the LMS submission needs.
