"""api_check: run a Postman v2.1 collection against a generated FastAPI app and report per-request results."""

from __future__ import annotations

import argparse
import ast
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from pydantic import BaseModel

DEFAULT_COLLECTION = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "demo_api.postman_collection.json"
)
_LOG_TAIL_LINES = 20


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


def load_collection(path: Path) -> list[dict]:
    """Return the collection's top-level items."""
    return json.loads(Path(path).read_text())["item"]


def find_app(app_dir: Path) -> str | None:
    """Return "module:app" for the first non-test .py file assigning `app = FastAPI(...)` at top level."""
    for path in sorted(Path(app_dir).glob("*.py")):
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            continue
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in tree.body:
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
                continue
            func = node.value.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == "FastAPI" and any(isinstance(t, ast.Name) and t.id == "app" for t in node.targets):
                return f"{path.stem}:app"
    return None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def _stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _send(client: httpx.Client, base: str, item: dict) -> CheckResult:
    name = item["name"]
    req = item["request"]
    example = item["response"][0]
    path = "/" + "/".join(req["url"].get("path", []))
    headers = {h["key"]: h["value"] for h in req.get("header", []) if not h.get("disabled")}
    body = req.get("body") or {}
    content = body.get("raw").encode() if body.get("mode") == "raw" else None
    try:
        resp = client.request(req["method"], base + path, headers=headers, content=content)
    except httpx.HTTPError as exc:
        return CheckResult(name=name, passed=False, detail=f"request failed: {exc!r}")
    if resp.status_code != example["code"]:
        return CheckResult(name=name, passed=False, detail=f"expected status {example['code']}, got {resp.status_code}")
    if example.get("body", "").strip():
        try:
            actual = resp.json()
        except ValueError:
            return CheckResult(name=name, passed=False, detail=f"response is not JSON: {resp.text[:200]!r}")
        expected = json.loads(example["body"])
        if actual != expected:
            return CheckResult(name=name, passed=False, detail=f"expected body {expected!r}, got {actual!r}")
    return CheckResult(name=name, passed=True)


def check_api(
    app_spec: str,
    app_dir: Path,
    collection: Path = DEFAULT_COLLECTION,
    *,
    startup_timeout: float = 10.0,
) -> list[CheckResult]:
    """Start the app under uvicorn, send each collection request, and compare to the saved example."""
    items = load_collection(collection)
    port = _free_port()
    failure: str | None = None
    results: list[CheckResult] = []
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", app_spec, "--app-dir", str(app_dir),
             "--host", "127.0.0.1", "--port", str(port)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + startup_timeout
            while not _port_open(port):
                if proc.poll() is not None:
                    failure = f"server exited with code {proc.returncode} before binding port"
                    break
                if time.monotonic() >= deadline:
                    failure = f"server timed out after {startup_timeout:g}s without binding port"
                    break
                time.sleep(0.1)
            if failure is None:
                base = f"http://127.0.0.1:{port}"
                with httpx.Client(timeout=5) as client:
                    results = [_send(client, base, item) for item in items]
        finally:
            _stop(proc)
        if failure is not None:
            log.seek(0)
            tail = "\n".join(log.read().decode("utf-8", errors="replace").splitlines()[-_LOG_TAIL_LINES:])
            return [CheckResult(name="startup", passed=False, detail=f"{failure}\n{tail}")]
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Postman fixture against a FastAPI app.")
    parser.add_argument("--app", help="module:attr of the app (default: auto-detected in --app-dir)")
    parser.add_argument("--app-dir", type=Path, default=Path("workspace"))
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    args = parser.parse_args(argv)
    app_spec = args.app or find_app(args.app_dir)
    if app_spec is None:
        print(f"FAIL startup: no `app = FastAPI(...)` found in {args.app_dir}")
        return 1
    results = check_api(app_spec, args.app_dir, args.collection)
    for r in results:
        print(f"PASS {r.name}" if r.passed else f"FAIL {r.name}: {r.detail}")
    return 0 if results and all(r.passed for r in results) else 1
