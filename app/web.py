"""`coder-web`: a local, read-only mini-IDE that streams a Coder run over SSE."""

from __future__ import annotations

import os
import queue
import threading
from collections.abc import Generator, AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from agents.coder_agent import CoderAgent
from agents.events import StepEvent
from tools.file_tools import read_file
from tools.workspace import RunLock, ToolError, WorkspaceBusyError, WorkspaceError

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


class RunRequest(BaseModel):
    task: str = Field(min_length=1)


def _frame(event: StepEvent) -> str:
    return f"data: {event.model_dump_json()}\n\n"


_DONE = object()


def _pump(events: Generator[StepEvent, None, None], out: queue.Queue, stop: threading.Event) -> None:
    """Drive the agent's stream on one thread; stop and close it once the client has gone."""
    try:
        for event in events:
            out.put(event)
            if stop.is_set():
                break
    finally:
        events.close()  # runs the stream's cleanup on this thread: releases the workspace lock
        out.put(_DONE)


async def _sse_body(first: StepEvent, events: Generator[StepEvent, None, None]) -> AsyncIterator[str]:
    out: queue.Queue = queue.Queue()
    stop = threading.Event()
    threading.Thread(target=_pump, args=(events, out, stop), daemon=True).start()
    try:
        yield _frame(first)
        while (event := await run_in_threadpool(out.get)) is not _DONE:
            yield _frame(event)
    finally:
        # Client disconnected (or the run ended): the pump stops at the next event boundary
        stop.set()


def create_app(agent: CoderAgent) -> FastAPI:
    app = FastAPI(title="Coder agent")
    # /api/run executes code: refuse foreign Host headers so a DNS-rebinding page can't drive it
    hosts = os.environ.get("CODER_WEB_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[h.strip() for h in hosts if h.strip()])
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/status")
    def status() -> dict:
        # Read the holder's marker; probing the lock itself could make a starting CLI run see "busy"
        return {"running": RunLock.is_running(agent.settings.state_dir)}

    @app.get("/api/files")
    def files() -> dict:
        ws = agent.workspace
        return {"files": [
            {"path": p, "size": (ws.root / p).stat().st_size, "touched": p in agent.last_written}
            for p in ws.list_files()
        ]}

    @app.get("/api/file")
    def file(path: str) -> dict:
        try:
            return {"path": path, "content": read_file(agent.workspace, path)}
        except WorkspaceError as e:
            raise HTTPException(400, str(e)) from None
        except ToolError as e:
            raise HTTPException(404, str(e)) from None

    @app.post("/api/run")
    async def run(req: RunRequest) -> StreamingResponse:
        events = agent.stream(req.task)
        try:
            # Pull the first event now so a busy workspace becomes a 409 instead of a broken stream
            first = await run_in_threadpool(next, events)
        except WorkspaceBusyError:
            raise HTTPException(409, "workspace busy") from None

        return StreamingResponse(_sse_body(first, events), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def main() -> None:
    import uvicorn

    host = os.environ.get("CODER_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("CODER_WEB_PORT", "8000"))
    uvicorn.run(create_app(CoderAgent()), host=host, port=port)
