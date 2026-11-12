import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_app_is_fastapi_instance():
    assert isinstance(app, FastAPI)


def test_get_health_returns_200_and_status_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_post_health_returns_405(client):
    response = client.post("/health")
    assert response.status_code == 405


def test_delivered_tests_pass_with_pytest():
    test_file = Path(__file__).parent / "test_main.py"
    if not test_file.exists():
        pytest.skip("delivered test_main.py not found next to this module")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(test_file)],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent),
    )
    assert result.returncode == 0, result.stdout + result.stderr
