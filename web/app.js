"use strict";
// Read-only mini-IDE: streams StepEvents from POST /api/run (fetch + ReadableStream, since
// EventSource can't POST) and mirrors the workspace, the last execution and the agent's steps.

const $ = (id) => document.getElementById(id);
const state = { running: false, selected: null, follow: true };

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function setRunning(running) {
  state.running = running;
  $("run").disabled = running;
  $("run").textContent = running ? "Running…" : "Run";
}

async function refreshFiles() {
  const res = await fetch("/api/files");
  const { files } = await res.json();
  const tree = $("tree");
  tree.replaceChildren();
  if (!files.length) tree.append(el("li", "empty", "No files yet"));
  for (const f of files) {
    const li = el("li", "file" + (f.touched ? " touched" : "") + (f.path === state.selected ? " selected" : ""), f.path);
    li.title = `${f.size} bytes` + (f.touched ? " (written this run)" : "");
    li.onclick = () => { state.follow = false; selectFile(f.path); };
    tree.append(li);
  }
}

async function selectFile(path) {
  state.selected = path;
  const res = await fetch("/api/file?" + new URLSearchParams({ path }));
  const code = $("code");
  $("file-name").textContent = path;
  if (!res.ok) { code.textContent = (await res.json()).detail; return; }
  const { content } = await res.json();
  code.textContent = content;
  code.className = path.endsWith(".py") ? "language-python" : "plaintext";
  delete code.dataset.highlighted;
  if (window.hljs && path.endsWith(".py")) hljs.highlightElement(code);
  for (const li of $("tree").children) li.classList.toggle("selected", li.textContent === path);
}

function addStep(kind, title, body) {
  const li = el("li", "step " + kind);
  li.append(el("div", "step-title", title));
  if (body) li.append(el("pre", "step-body", body));
  $("timeline").append(li);
  li.scrollIntoView({ block: "end", behavior: "smooth" });
}

function shortArgs(args) {
  return Object.entries(args || {})
    .map(([k, v]) => {
      let s = typeof v === "string" ? v : JSON.stringify(v);
      s = s.replace(/\n/g, "⏎");
      return `${k}=${s.length > 60 ? s.slice(0, 59) + "…" : s}`;
    })
    .join(", ");
}

function renderResult(r) {
  const box = $("result");
  box.replaceChildren();
  box.hidden = false;
  box.className = "result " + r.status;
  const head = el("div", "result-head");
  head.append(el("span", "badge " + r.status, r.status.replace("_", " ")));
  head.append(el("span", "verified " + (r.verified ? "yes" : "no"), r.verified ? "verified ✓" : "not verified ✗"));
  box.append(head);
  if (r.verification_notes.length) box.append(el("p", "notes", r.verification_notes.join("; ")));
  if (r.error) box.append(el("p", "notes", r.error));
  box.append(el("h3", null, "Explanation"), el("p", null, r.explanation));
  box.append(el("h3", null, "Plan"));
  const ol = el("ol");
  r.plan.forEach((p) => ol.append(el("li", null, p)));
  box.append(ol);
  const ran = r.result.executed
    ? `${r.result.target}: exit ${r.result.exit_code}${r.result.timed_out ? " (timed out)" : ""}, ${r.result.duration_s.toFixed(2)}s`
    : "code was never executed";
  box.append(el("h3", null, "Last execution"), el("p", "mono", ran));
  if (r.memories_used.length) {
    box.append(el("h3", null, "Memories used"));
    const ul = el("ul", "memories");
    r.memories_used.forEach((m) => ul.append(el("li", null, `(${m.distance.toFixed(2)}) ${m.text.split("\n")[0]}`)));
    box.append(ul);
  }
  box.scrollIntoView({ block: "end", behavior: "smooth" });
}

async function handle(ev) {
  const d = ev.data;
  $("iter").textContent = ev.iteration;
  $("max").textContent = ev.max_iterations;
  switch (ev.kind) {
    case "start":
      addStep("start", "Task", d.task);
      break;
    case "memory": {
      const hits = d.hits || [];
      const lines = hits.map((h) => `(${h.distance.toFixed(2)}) ${h.text.split("\n")[0]}`);
      addStep("memory", `Memory: ${hits.length} past run(s), ${d.window_turns} recent turn(s)`, lines.join("\n") || null);
      break;
    }
    case "thought":
      addStep("thought", "Thought", d.text);
      break;
    case "plan":
      addStep("plan", "Plan", d.steps.map((s, i) => `${i + 1}. ${s}`).join("\n"));
      break;
    case "action":
      addStep("action", `Action: ${d.tool}`, shortArgs(d.args));
      break;
    case "observation": {
      const lines = d.text.split("\n");
      addStep(d.ok ? "observation" : "observation bad", `Observation: ${d.tool}`,
        lines.slice(0, 12).join("\n") + (lines.length > 12 ? `\n… ${lines.length - 12} more lines` : ""));
      if (d.tool === "exec_python") $("terminal").textContent = d.text;
      await refreshFiles();
      if (d.tool === "write_file" && d.ok && state.follow) {
        const m = d.text.match(/^wrote (.+?) \(/);
        if (m) await selectFile(m[1]);
      }
      break;
    }
    case "nudge":
    case "limit":
    case "error":
      addStep(ev.kind, ev.kind[0].toUpperCase() + ev.kind.slice(1), d.text || d.message ||
        ["not executed: " + ((d.unexecuted || []).join(", ") || "nothing"), ...(d.rejected || [])].join("\n"));
      break;
    case "result":
      renderResult(d.result);
      break;
  }
}

async function run(task) {
  setRunning(true);
  state.follow = true;
  $("timeline").replaceChildren();
  $("result").hidden = true;
  $("terminal").textContent = "Waiting for exec_python…";
  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task }),
    });
    if (!res.ok) {
      const detail = res.status === 409 ? "Workspace busy: another Coder run is in progress." : (await res.text());
      addStep("error", "Could not start", detail);
      return;
    }
    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += value;
      let cut;
      while ((cut = buf.indexOf("\n\n")) >= 0) {
        const frame = buf.slice(0, cut);
        buf = buf.slice(cut + 2);
        if (frame.startsWith("data: ")) await handle(JSON.parse(frame.slice(6)));
      }
    }
  } catch (err) {
    addStep("error", "Connection lost", String(err));
  } finally {
    setRunning(false);
    refreshFiles();
  }
}

$("run-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const task = $("task").value.trim();
  if (task && !state.running) run(task);
});

// Another process (e.g. the CLI) may hold the workspace: keep Run disabled while it does
setInterval(async () => {
  if (state.running) return;
  try {
    const { running } = await (await fetch("/api/status")).json();
    $("run").disabled = running;
    $("run").title = running ? "Another Coder run is using the workspace" : "";
  } catch { /* server restarting */ }
}, 3000);

refreshFiles();
