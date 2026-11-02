"""Deliberately broken demo API: /echo accepts anything, so validation requests fail."""

from __future__ import annotations

from fastapi import FastAPI, Request

app = FastAPI()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/echo")
async def echo(request: Request) -> dict:
    return {}
