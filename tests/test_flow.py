"""End-to-end patient journeys through the web app."""
from datetime import timedelta

from triagecare.models import Appointment, Patient, TriageAssessment, Visit, VisitEvent
from triagecare.timeutil import to_local, utcnow

STABLE = dict(heart_rate="80", resp_rate="16", spo2="98", systolic_bp="124", diastolic_bp="80",
              temp_c="36.9", blood_glucose="", gcs_score="15", pain_score="2")
SICK = dict(heart_rate="132", resp_rate="31", spo2="86", systolic_bp="84", diastolic_bp="50",
            temp_c="38.9", blood_glucose="", gcs_score="13", pain_score="7")


def register(client, csrf, **extra):
    data = dict(first_name="Amna", last_name="Baloch", date_of_birth="1970-02-11", gender="Female",
                phone="0300-1234567", comorbidity="None", csrf_token=csrf)
    data.update(extra)
    return client.post("/patients/new", data=data, follow_redirects=False)


def arrive(client, csrf, patient_id, complaint="Chest pain"):
    return client.post("/visits/new", data={"patient_id": patient_id, "chief_complaint": complaint,
                                           "arrival_mode": "walk_in", "csrf_token": csrf}, follow_redirects=False)


def triage(client, csrf, visit_id, vitals):
    return client.post(f"/visits/{visit_id}/triage", data={**vitals, "csrf_token": csrf}, follow_redirects=False)


def test_register_patient_assigns_mrn(client, login, db_session):
    csrf = login("reception")
    r = register(client, csrf)
    assert r.status_code == 303
    p = db_session.query(Patient).one()
    assert p.mrn == "TC-000001"
    assert r.headers["location"] == f"/patients/{p.id}"


def test_register_validation_errors_are_shown(client, login):
    csrf = login("reception")
    r = register(client, csrf, first_name="", date_of_birth="2999-01-01", cnic="123")
    assert r.status_code == 422
    assert "Required" in r.text
    assert "can&#39;t be in the future" in r.text or "can't be in the future" in r.text
    assert "Invalid format" in r.text


def test_full_journey(client, login, db_session, patient):
    # Nurse registers the arrival and triages a very sick patient
    csrf = login("nurse")
    r = arrive(client, csrf, patient.id)
    visit = db_session.query(Visit).one()
    assert r.headers["location"] == f"/visits/{visit.id}/triage"

    r = triage(client, csrf, visit.id, SICK)
    assert r.status_code == 303
    db_session.refresh(visit)
    a = db_session.query(TriageAssessment).one()
    assert visit.status == "triaged"
    assert visit.tier == "high"
    assert a.final_tier == "high"
    assert visit.ward.level == "resus"
    assert visit.doctor is not None
    assert any(f["level"] == "red" for f in a.red_flags)

    # It shows up first in the queue and in the API
    queue = client.get("/api/v1/queue").json()
    assert queue[0]["visit_id"] == visit.id
    assert queue[0]["tier"] == "high"

    # A doctor picks the patient up, changes priority, and closes the visit
    csrf = login("dr.hamid")
    assert client.post(f"/visits/{visit.id}/start", data={"csrf_token": csrf}).status_code == 200
    db_session.refresh(visit)
    assert visit.status == "in_treatment"
    assert visit.doctor.username == "dr.hamid"

    client.post(f"/visits/{visit.id}/override", data={"tier": "medium", "reason": "Improved after oxygen", "csrf_token": csrf})
    db_session.refresh(visit)
    db_session.refresh(a)
    assert visit.tier == "medium"
    assert a.override_reason == "Improved after oxygen"

    client.post(f"/visits/{visit.id}/complete", data={"disposition": "admitted", "doctor_notes": "Pneumonia", "csrf_token": csrf})
    db_session.refresh(visit)
    assert visit.status == "completed"
    assert visit.disposition == "admitted"

    kinds = [e.kind for e in db_session.query(VisitEvent).order_by(VisitEvent.id)]
    assert kinds == ["arrival", "triage", "seen", "override", "complete"]


def test_queue_orders_by_risk_then_wait(client, login, db_session):
    csrf = login("nurse")
    ids = []
    for name, vitals in (("Early", STABLE), ("Late", SICK)):
        register(client, csrf, first_name=name)
        pid = db_session.query(Patient).filter_by(first_name=name).one().id
        arrive(client, csrf, pid)
        vid = db_session.query(Visit).filter_by(patient_id=pid).one().id
        triage(client, csrf, vid, vitals)
        ids.append(vid)
    # The stable patient arrived first but the sick one should be seen first.
    early = db_session.get(Visit, ids[0])
    early.arrived_at = utcnow() - timedelta(hours=1)
    db_session.commit()
    order = [i["visit_id"] for i in client.get("/api/v1/queue").json()]
    assert order == [ids[1], ids[0]]


def test_cannot_open_two_visits_for_one_patient(client, login, db_session, patient):
    csrf = login("reception")
    arrive(client, csrf, patient.id)
    r = arrive(client, csrf, patient.id)
    assert db_session.query(Visit).count() == 1
    assert r.headers["location"].startswith("/visits/")


def test_live_preview_saves_nothing(client, login, db_session, patient):
    csrf = login("nurse")
    arrive(client, csrf, patient.id)
    visit = db_session.query(Visit).one()
    r = client.post(f"/visits/{visit.id}/triage/preview", data={**SICK, "csrf_token": csrf})
    assert "High risk" in r.text
    assert db_session.query(TriageAssessment).count() == 0

    r = client.post(f"/visits/{visit.id}/triage/preview", data={"heart_rate": "90", "csrf_token": csrf})
    assert "Waiting for" in r.text


def test_appointment_booking_and_check_in(client, login, db_session, patient):
    csrf = login("reception")
    when = (to_local(utcnow()) + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M")
    r = client.post("/appointments/new", data={"patient_id": patient.id, "scheduled_at": when,
                                               "reason": "BP review", "doctor_id": "", "csrf_token": csrf},
                    follow_redirects=False)
    assert r.status_code == 303
    appt = db_session.query(Appointment).one()
    assert appt.status == "booked"

    client.post(f"/appointments/{appt.id}/check-in", data={"csrf_token": csrf})
    db_session.refresh(appt)
    assert appt.status == "checked_in"
    visit = db_session.query(Visit).one()
    assert visit.appointment_id == appt.id
    assert visit.arrival_mode == "appointment"


def test_appointment_in_the_past_is_rejected(client, login, patient):
    csrf = login("reception")
    when = (to_local(utcnow()) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M")
    r = client.post("/appointments/new", data={"patient_id": patient.id, "scheduled_at": when,
                                               "reason": "BP review", "csrf_token": csrf})
    assert r.status_code == 422
    assert "already passed" in r.text


def test_api_predict(client, login):
    login("nurse")
    body = {"age": 70, "gender": "Male", "comorbidity": "Cardiovascular", "heart_rate": 130,
            "resp_rate": 28, "systolic_bp": 85, "spo2": 88, "temp_c": 37.2, "gcs_score": 15, "pain_score": 8}
    r = client.post("/api/v1/predict", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["tier"] == "high"
    assert data["recommended_ward"] == "Resuscitation Bay"
    assert len(data["drivers"]) == 10  # glucose not given, so not explained


def test_api_predict_validates_ranges(client, login):
    login("nurse")
    r = client.post("/api/v1/predict", json={"age": 30, "gender": "Male", "heart_rate": 900, "resp_rate": 16,
                                            "spo2": 98, "temp_c": 37, "gcs_score": 15, "pain_score": 0})
    assert r.status_code == 422


def test_every_page_renders(client, login, db_session, patient):
    csrf = login("admin")
    arrive(client, csrf, patient.id)
    visit = db_session.query(Visit).one()
    triage(client, csrf, visit.id, STABLE)
    for url in ["/", "/queue", "/wards", "/patients", f"/patients/{patient.id}", "/patients/new",
                "/appointments", "/appointments/new", "/visits/new", f"/visits/{visit.id}",
                f"/visits/{visit.id}/triage", "/admin/users", "/admin/model"]:
        r = client.get(url)
        assert r.status_code == 200, url
