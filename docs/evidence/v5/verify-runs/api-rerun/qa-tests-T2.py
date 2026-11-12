import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_echo_nested_object_returns_identical_json(client):
    body = {
        "user": {"name": "Ada", "tags": ["x", "y"], "meta": {"depth": {"level": 3}}},
        "active": True,
        "score": 1.5,
        "nothing": None,
        "items": [1, "two", None, {"k": []}],
    }
    response = client.post("/echo", json=body)
    assert response.status_code == 200
    assert response.json() == body


def test_echo_empty_object_returns_empty_object(client):
    response = client.post("/echo", json={})
    assert response.status_code == 200
    assert response.json() == {}


def test_get_echo_returns_405(client):
    response = client.get("/echo")
    assert response.status_code == 405


def test_delivered_tests_pass_with_pytest(tmp_path):
    # Runs the Coder's delivered test module in a subprocess; must exit 0.
    here = Path(__file__).resolve().parent
    candidates = [p for p in here.glob("test_*.py") if not p.name.startswith("test_qa_")]
    assert candidates, "delivered test file not found"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(candidates[0])],
        cwd=str(here),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
