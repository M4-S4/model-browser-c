import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import router
from src.plugin_api import AccountType, User, get_current_active_user

HEALTH_URL = "/api/plugins/model-browser-c/health"


def _user() -> User:
    return User(
        id="user-1",
        username="user",
        email="user@test.com",
        password_hash="hash",
        account_type=AccountType.USER,
    )


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router)
    yield app
    app.dependency_overrides.clear()


def test_health_answers_any_signed_in_user(app):
    app.dependency_overrides[get_current_active_user] = _user

    with TestClient(app) as client:
        response = client.get(HEALTH_URL)

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "data": {"status": "ok", "plugin": "model-browser-c", "version": "0.1.0"},
    }


def test_health_needs_a_signed_in_user(app):
    with TestClient(app) as client:
        response = client.get(HEALTH_URL)

    assert response.status_code in (401, 403)
