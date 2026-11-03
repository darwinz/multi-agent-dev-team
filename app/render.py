"""rich renderers for step events and the final CoderResult."""

from __future__ import annotations

import json

from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from agents.events import StepEvent
from agents.result import CoderResult

_STATUS_STYLE = {"success": "bold green", "failed": "bold red", "iteration_limit": "bold yellow", "error": "bold red"}


def _short(value: object, limit: int = 80) -> str:
    s = value if isinstance(value, str) else json.dumps(value)
    s = s.replace("\n", "\\n")
    return s if len(s) <= limit else s[: limit - 1] + "…"


def render_event(console: Console, event: StepEvent) -> None:
    d = event.data
    prefix = Text(f"[{event.iteration}/{event.max_iterations}] ", style="bold cyan")
    k = event.kind
    if k == "start":
        console.rule(Text(f"Task ({d.get('model')}, run {d.get('run_id')})"))
        console.print(Text(d.get("task", ""), style="bold"))
    elif k == "memory":
        hits = d.get("hits", [])
        line = f"memory: {len(hits)} relevant past run(s), {d.get('window_turns', 0)} recent turn(s)"
        if hits:
            line += f"; top: {hits[0]['text'].splitlines()[0]} (distance {hits[0]['distance']:.2f})"
        console.print(prefix + Text(line, style="magenta"))
    elif k == "thought":
        console.print(prefix + Text("thought: ", style="bold") + Text(d.get("text", ""), style="dim"))
    elif k == "plan":
        console.print(prefix + Text("plan:", style="bold"))
        for i, step in enumerate(d.get("steps", []), 1):
            console.print(Text(f"    {i}. {step}"))
    elif k == "action":
        args = ", ".join(f"{key}={_short(v)}" for key, v in (d.get("args") or {}).items())
        console.print(prefix + Text(f"action: {d.get('tool')}({args})", style="bold blue"))
    elif k == "observation":
        lines = d.get("text", "").splitlines()
        body = "\n".join(lines[:15]) + (f"\n… ({len(lines) - 15} more lines)" if len(lines) > 15 else "")
        style = "green" if d.get("ok") else "red"
        console.print(prefix + Text("observation:", style=f"bold {style}"))
        console.print(Text(body, style=style), soft_wrap=True)
    elif k in ("nudge", "limit", "error"):
        msg = {
            "nudge": d.get("text", ""),
            "limit": "iteration limit reached; not executed: " + (", ".join(d.get("unexecuted", [])) or "nothing")
            + "".join(f"; {r}" for r in d.get("rejected", [])),
            "error": d.get("message", ""),
        }[k]
        console.print(prefix + Text(f"{k}: {msg}", style="bold yellow" if k != "error" else "bold red"))


def render_result(console: Console, result: CoderResult) -> None:
    r = result.result
    parts: list = [
        Text.assemble(("status: ", "bold"), (result.status, _STATUS_STYLE[result.status]),
                      f"   iterations: {result.iterations}   run: {result.run_id}"),
        Text.assemble(("verified: ", "bold"), ("✓" if result.verified else "✗", "green" if result.verified else "red"),
                      ("  " + "; ".join(result.verification_notes)) if result.verification_notes else ""),
        Text("plan:", style="bold"),
        *[Text(f"  {i}. {step}") for i, step in enumerate(result.plan, 1)],
        Text.assemble(("explanation: ", "bold"), result.explanation),
        Text.assemble(("files: ", "bold"), ", ".join(result.files) or "none"),
    ]
    if r.executed:
        tail = "\n".join(r.stdout.strip().splitlines()[-8:])
        parts.append(Text.assemble(("last execution: ", "bold"),
                                   f"{r.target}  exit_code={r.exit_code}  timed_out={r.timed_out}  {r.duration_s:.2f}s"))
        if tail:
            parts.append(Text(tail, style="dim"))
    else:
        parts.append(Text("last execution: code was never executed", style="red"))
    if result.code:
        lines = result.code.splitlines()
        parts.append(Syntax("\n".join(lines[:60]), "python", line_numbers=True))
        if len(lines) > 60:
            parts.append(Text(f"… {len(lines) - 60} more lines", style="dim"))
    if result.memories_used:
        parts.append(Text("memories used:", style="bold"))
        parts += [Text(f"  ({m.distance:.2f}) {m.text.splitlines()[0]}", style="magenta") for m in result.memories_used]
    if result.error:
        parts.append(Text.assemble(("error: ", "bold red"), result.error))
    console.print(Panel(Group(*parts), title="CoderResult", border_style=_STATUS_STYLE[result.status].split()[-1]))


# --- team (PM -> Coder <-> QA) renderers --------------------------------------------

_TASK_STYLE = {"done": "green", "failed": "red", "in_progress": "yellow", "pending": "dim"}
_RUN_STYLE = {"completed": "bold green", "partial": "bold yellow", "failed": "bold red", "running": "bold"}


def render_spec(console: Console, data: dict) -> None:
    if "error" in data and "name" not in data:
        console.print(Panel(Text(str(data["error"]), style="red"), title="PM failed", border_style="red"))
        return
    table = Table("id", "priority", "title", "covers", expand=True)
    for t in data.get("tasks", []):
        table.add_row(t["task_id"], t["priority"], t["title"], ", ".join(t["covers"]))
    criteria = [Text(f"  {c['id']}: {c['text']}") for c in data.get("acceptance_criteria", [])]
    body = Group(Text(data.get("description", "")), Text("acceptance criteria:", style="bold"), *criteria, table)
    console.print(Panel(body, title=f"Spec: {data.get('name', '')}", border_style="cyan"))


def render_consolidation(console: Console, data: dict) -> None:
    titles = {t["task_id"]: t["title"] for t in data.get("original_tasks", [])}
    lines = [
        Text(f"{new} <- " + ", ".join(f"{o} ({titles.get(o, '?')})" for o in originals))
        for new, originals in data.get("merged_from", {}).items()
    ]
    n = len(data.get("original_tasks", []))
    console.print(Panel(Group(*lines), title=f"Consolidated {n} tasks ({data.get('method')})",
                        border_style="magenta"))


def render_task_header(console: Console, data: dict) -> None:
    console.rule(Text(f"T{data['position']}/{data['total']} · {data['priority']} · {data['title']}",
                      style="bold"), align="left")


def _n(count: int, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


def render_a2a(console: Console, record: dict) -> None:
    """One line per broker record, e.g. `A2A coder → qa review_request #1a2b3c4d round 1 · 2 files`."""
    m = record["message"]
    p = m["payload"]
    tag = f"{m['intent']} #{m['correlation_id'][:8]}"
    if record["event"] == "received":
        console.print(Text(f"A2A {m['receiver']} received {tag}", style="dim"), soft_wrap=True)
        return
    if record["event"] == "rejected":
        console.print(Text(f"A2A {m['receiver']} rejected {tag} from {m['sender']}: {record['reason']}",
                           style="bold red"), soft_wrap=True)
        return
    detail = {
        "review_request": lambda: _n(len(p["files"]), "file"),
        "fix_instruction": lambda: _n(len(p["issues"]), "issue"),
        "approved": lambda: _n(p["tests_run"], "test") + " passed",
        "final_report": lambda: f"{p['reason']} · {len(p['unresolved'])} unresolved",
    }[m["intent"]]()
    style = {"approved": "bold green", "final_report": "bold red", "fix_instruction": "yellow"}.get(m["intent"], "blue")
    console.print(Text(f"A2A {m['sender']} → {m['receiver']} {tag} round {p['round']} · {detail}", style=style),
                  soft_wrap=True)


def render_reflection(console: Console, data: dict) -> None:
    items = (data.get("critique") or {}).get("items", [])
    if data.get("error"):
        line = f"self-review: no revision ({data['error']})"
        style = "bold red"
    else:
        line = f"self-review: {_n(len(items), 'critique item')}, revised {', '.join(data['changed'])}"
        style = "magenta"
    console.print(Text(line, style=style), soft_wrap=True)
    for item in items:
        console.print(Text(f"    {item['kind']} {item['location']}: {item['concern']}", style="dim"), soft_wrap=True)
    if data.get("dropped"):
        console.print(Text(f"    dropped (no longer on disk): {', '.join(data['dropped'])}", style="yellow"),
                      soft_wrap=True)


def render_qa_report(console: Console, data: dict) -> None:
    head = f"QA round {data['round']}: "
    if data.get("qa_error"):
        console.print(Text(head + f"QA error: {data['qa_error']}", style="bold red"), soft_wrap=True)
    else:
        verdict = "approved" if data["approved"] else _n(len(data["issues"]), "issue")
        console.print(Text(head + f"{data['passed']}/{data['tests_run']} passed in {data['test_file']} · {verdict}"
                           + (" · workspace suite restored (tampered)" if data.get("tampered") else ""),
                           style="bold green" if data["approved"] else "bold yellow"), soft_wrap=True)
    for issue in data.get("issues", []):
        first = issue["assertion"].splitlines()[0] if issue["assertion"] else ""
        console.print(Text(f"    {issue['severity']} {issue['test'] or issue['location']}: {_short(first, 100)}",
                           style="red"), soft_wrap=True)


def render_team_summary(console: Console, data: dict) -> None:
    table = Table("id", "status", "qa rounds", "qa", "iterations", "files")
    # One row per task that executes (the merged tasks after consolidation)
    records = data.get("task_status") or {}
    for task_id in (t["task_id"] for t in data.get("tasks") or []):
        rec = records.get(task_id, {"status": "pending"})
        table.add_row(task_id, Text(rec["status"], style=_TASK_STYLE.get(rec["status"], "")),
                      str(rec.get("qa_rounds", 0)), rec.get("qa_outcome") or "-",
                      str(rec.get("iterations", 0)), ", ".join(rec.get("files", [])) or "none")
    status = data.get("status", "failed")
    parts: list = [table, Text.assemble(("run status: ", "bold"), (status, _RUN_STYLE.get(status, "bold")))]
    for task_id in (t["task_id"] for t in data.get("tasks") or []):
        for line in records.get(task_id, {}).get("unresolved", []):
            parts.append(Text(f"{task_id} unresolved: {_short(line, 120)}", style="red"))
    if data.get("error"):
        parts.append(Text.assemble(("error: ", "bold red"), str(data["error"])))
    console.print(Panel(Group(*parts), title="Team summary", border_style=_RUN_STYLE.get(status, "bold").split()[-1]))
