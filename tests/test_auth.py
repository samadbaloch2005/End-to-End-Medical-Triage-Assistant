def test_pages_require_login(client):
    r = client.get("/queue", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login?next=/queue"


def test_api_returns_401_when_signed_out(client):
    assert client.get("/api/v1/queue").status_code == 401


def test_wrong_password_is_rejected(client):
    from tests.conftest import csrf_of

    r = client.post("/login", data={"username": "nurse", "password": "nope", "csrf_token": csrf_of(client)})
    assert r.status_code == 400
    assert "match" in r.text


def test_post_without_csrf_token_is_refused(client, login):
    login("reception")
    r = client.post("/patients/new", data={"first_name": "A"})
    assert r.status_code == 403


def test_login_does_not_redirect_offsite(client):
    from tests.conftest import csrf_of

    r = client.post(
        "/login",
        data={"username": "nurse", "password": "password123", "next": "//evil.example", "csrf_token": csrf_of(client)},
        follow_redirects=False,
    )
    assert r.headers["location"] == "/"


def test_receptionist_cannot_triage(client, login, db_session, patient):
    from triagecare.models import Visit

    visit = Visit(patient_id=patient.id, chief_complaint="Cough")
    db_session.add(visit)
    db_session.commit()
    login("reception")
    assert client.get(f"/visits/{visit.id}/triage").status_code == 403


def test_only_admin_sees_staff_page(client, login):
    login("nurse")
    assert client.get("/admin/users").status_code == 403
    login("admin")
    assert client.get("/admin/users").status_code == 200


def test_deactivated_user_is_signed_out(client, login, db_session):
    from triagecare.models import User

    login("nurse")
    nurse = db_session.query(User).filter_by(username="nurse").one()
    nurse.is_active = False
    db_session.commit()
    assert client.get("/", follow_redirects=False).status_code == 303
