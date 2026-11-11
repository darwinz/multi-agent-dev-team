"""Reference demo API: passes every request in the Postman fixture."""

from __future__ import annotations

from typing import Annotated

from fastapi import Body, FastAPI

app = FastAPI()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/echo")
def echo(body: Annotated[dict, Body()]) -> dict:
    return body
