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


def _assert_422_with_detail(response):
    assert response.status_code == 422
    body = response.json()
    assert isinstance(body, dict)
    assert "detail" in body


def test_malformed_json_returns_422_with_detail(client):
    response = client.post(
        "/echo", content=b"{invalid", headers={"content-type": "application/json"}
    )
    _assert_422_with_detail(response)


@pytest.mark.parametrize("raw", [b"[1,2]", b'"x"', b"5", b"true", b"null"])
def test_non_object_json_returns_422(client, raw):
    response = client.post(
        "/echo", content=raw, headers={"content-type": "application/json"}
    )
    _assert_422_with_detail(response)


def test_empty_body_returns_422_with_detail(client):
    response = client.post("/echo", content=b"")
    _assert_422_with_detail(response)


def test_missing_body_returns_422_with_detail(client):
    response = client.post("/echo")
    _assert_422_with_detail(response)


@pytest.mark.parametrize(
    "body",
    [{"hello": "world"}, {"n": [1, None, {"k": True}]}, {}],
)
def test_valid_object_body_still_echoed(client, body):
    response = client.post("/echo", json=body)
    assert response.status_code == 200
    assert response.json() == body


def test_tests_pass_with_pytest(tmp_path):
    # Runs the Coder's delivered test file in a subprocess and checks it passes.
    delivered = Path(__file__).resolve().parent / "test_echo_validation.py"
    assert delivered.exists()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(delivered)],
        cwd=str(Path(__file__).resolve().parent),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
