"""`coder-web`: a local, read-only mini-IDE that streams a Coder run over SSE."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from agents.coder_agent import CoderAgent
from agents.events import StepEvent
from tools.file_tools import read_file
from tools.workspace import RunLock, ToolError, WorkspaceBusyError, WorkspaceError

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


class RunRequest(BaseModel):
    task: str = Field(min_length=1)


def _frame(event: StepEvent) -> str:
    return f"data: {event.model_dump_json()}\n\n"


def create_app(agent: CoderAgent) -> FastAPI:
    app = FastAPI(title="Coder agent")
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/status")
    def status() -> dict:
        return {"running": RunLock(agent.settings.state_dir).is_locked_elsewhere()}

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
        events: Iterator[StepEvent] = agent.stream(req.task)
        try:
            # Pull the first event now so a busy workspace becomes a 409 instead of a broken stream
            first = await run_in_threadpool(next, events)
        except WorkspaceBusyError:
            raise HTTPException(409, "workspace busy") from None

        async def body() -> AsyncIterator[str]:
            try:
                yield _frame(first)
                async for event in iterate_in_threadpool(events):
                    yield _frame(event)
            finally:
                # Browser went away mid-run: close the generator so its lock is released
                try:
                    await run_in_threadpool(events.close)
                except ValueError:  # still executing in another thread; it finishes on its own
                    pass

        return StreamingResponse(body(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def main() -> None:
    import uvicorn

    host = os.environ.get("CODER_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("CODER_WEB_PORT", "8000"))
    uvicorn.run(create_app(CoderAgent()), host=host, port=port)
