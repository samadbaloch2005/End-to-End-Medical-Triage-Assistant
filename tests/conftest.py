"""Test setup: a fresh SQLite database per test, seeded with staff and wards.

The trained model is needed for triage tests. If it isn't there yet, it is
trained once per session (about 15 seconds).
"""
import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret")

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    from triagecare.config import get_settings

    if not get_settings().model_path.exists():
        from triagecare.cli import train

        train()


@pytest.fixture()
def db_session(tmp_path, monkeypatch):
    from sqlalchemy.orm import sessionmaker

    from triagecare import db as db_module
    from triagecare.db import Base, make_engine
    from triagecare.models import Role, User, Ward
    from triagecare.security import hash_password
    from triagecare.seed import STAFF, WARDS

    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(db_module, "SessionLocal", TestSession)

    with TestSession() as s:
        for code, name, level, capacity, desc in WARDS:
            s.add(Ward(code=code, name=name, level=level, capacity=capacity, description=desc))
        pw = hash_password("password123")
        for username, name, role, specialty in STAFF:
            s.add(User(username=username, full_name=name, role=role, specialty=specialty, password_hash=pw))
        s.commit()

    session = TestSession()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture()
def client(db_session):
    from fastapi.testclient import TestClient

    from triagecare.db import get_db
    from triagecare.main import app

    def _get_db():
        yield db_session

    app.dependency_overrides[get_db] = _get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def csrf_of(client) -> str:
    html = client.get("/login").text
    return re.search(r'name="csrf-token" content="([^"]+)"', html).group(1)


@pytest.fixture()
def login(client):
    """login("nurse") signs the test client in and returns the CSRF token."""

    def _login(username: str) -> str:
        client.cookies.clear()
        token = csrf_of(client)
        r = client.post(
            "/login",
            data={"username": username, "password": "password123", "csrf_token": token},
            follow_redirects=False,
        )
        assert r.status_code == 303, r.text
        return csrf_of(client)

    return _login


@pytest.fixture()
def patient(db_session):
    from datetime import date

    from triagecare.models import Patient

    p = Patient(
        mrn="TC-000001",
        first_name="Test",
        last_name="Patient",
        date_of_birth=date(1960, 5, 1),
        gender="Male",
        comorbidity="Hypertension",
    )
    db_session.add(p)
    db_session.commit()
    return p
