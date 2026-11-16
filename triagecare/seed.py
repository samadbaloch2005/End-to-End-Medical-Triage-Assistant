"""Demo data: staff accounts, wards, patients, today's appointments and a busy queue.

Everything here is fictional.
"""
import random
from datetime import date, datetime, time, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from triagecare.config import get_settings
from triagecare.models import (
    Appointment,
    AppointmentStatus,
    ArrivalMode,
    Patient,
    Role,
    User,
    Visit,
    VisitStatus,
    Ward,
)
from triagecare.security import hash_password
from triagecare.services.triage import log_event, record_triage
from triagecare.timeutil import local_today, to_utc, utcnow

WARDS = [
    ("RES", "Resuscitation Bay", "resus", 4, "Immediately life-threatening. Continuous monitoring."),
    ("ACU", "Acute Care (Majors)", "acute", 10, "Serious illness needing urgent assessment and treatment."),
    ("URG", "Urgent Care", "urgent", 14, "Stable but needs to be seen soon. Observation chairs."),
    ("GEN", "General OPD / Minors", "general", 20, "Minor illness and injury, routine review."),
]

STAFF = [
    ("admin", "Sana Iqbal", Role.ADMIN, None),
    ("reception", "Bilal Ahmed", Role.RECEPTIONIST, None),
    ("nurse", "Ayesha Khan", Role.NURSE, None),
    ("nurse.fatima", "Fatima Raza", Role.NURSE, None),
    ("dr.hamid", "Hamid Qureshi", Role.DOCTOR, "Emergency Medicine"),
    ("dr.nadia", "Nadia Siddiqui", Role.DOCTOR, "Emergency Medicine"),
    ("dr.usman", "Usman Tariq", Role.DOCTOR, "Cardiology"),
    ("dr.mehwish", "Mehwish Ali", Role.DOCTOR, "Pulmonology"),
    ("dr.kamran", "Kamran Javed", Role.DOCTOR, "Neurology"),
    ("dr.saima", "Saima Haider", Role.DOCTOR, "Internal Medicine"),
    ("dr.omer", "Omer Farooq", Role.DOCTOR, "General Practice"),
]

FIRST_M = ["Ali", "Ahmed", "Hassan", "Imran", "Zain", "Faisal", "Asad", "Tariq", "Shahid", "Waqas", "Junaid", "Rashid", "Naveed", "Saad", "Hamza"]
FIRST_F = ["Sara", "Hina", "Maryam", "Zainab", "Rabia", "Amna", "Nida", "Sadia", "Iqra", "Khadija", "Mahnoor", "Samina", "Rukhsana", "Areeba", "Nazia"]
LAST = ["Khan", "Ahmed", "Malik", "Hussain", "Shah", "Butt", "Chaudhry", "Baloch", "Mirza", "Abbasi", "Sheikh", "Rana", "Akhtar", "Bhatti", "Memon", "Jamali", "Magsi"]
CITIES = ["Gulshan-e-Iqbal, Karachi", "Model Town, Lahore", "G-10, Islamabad", "Saddar, Rawalpindi", "Satellite Town, Quetta", "Hayatabad, Peshawar", "Latifabad, Hyderabad"]

# (complaint, arrival mode, vitals template, minutes ago)
SCENARIOS = [
    ("Crushing chest pain radiating to left arm", ArrivalMode.AMBULANCE,
     dict(heart_rate=128, resp_rate=26, systolic_bp=86, diastolic_bp=58, spo2=91, temp_c=36.9, blood_glucose=168, gcs_score=15, pain_score=9)),
    ("Shortness of breath, known asthmatic", ArrivalMode.WALK_IN,
     dict(heart_rate=118, resp_rate=31, systolic_bp=132, diastolic_bp=84, spo2=87, temp_c=37.4, blood_glucose=110, gcs_score=15, pain_score=4)),
    ("Found confused at home, family concerned", ArrivalMode.AMBULANCE,
     dict(heart_rate=96, resp_rate=20, systolic_bp=168, diastolic_bp=96, spo2=95, temp_c=37.1, blood_glucose=72, gcs_score=11, pain_score=2)),
    ("High fever and vomiting for two days", ArrivalMode.WALK_IN,
     dict(heart_rate=114, resp_rate=22, systolic_bp=104, diastolic_bp=66, spo2=95, temp_c=39.2, blood_glucose=132, gcs_score=15, pain_score=5)),
    ("Very high sugar reading on home meter, thirsty", ArrivalMode.WALK_IN,
     dict(heart_rate=104, resp_rate=21, systolic_bp=118, diastolic_bp=74, spo2=97, temp_c=37.3, blood_glucose=312, gcs_score=15, pain_score=3)),
    ("Severe abdominal pain, right side", ArrivalMode.WALK_IN,
     dict(heart_rate=102, resp_rate=19, systolic_bp=126, diastolic_bp=80, spo2=98, temp_c=38.1, blood_glucose=118, gcs_score=15, pain_score=8)),
    ("Twisted ankle playing cricket", ArrivalMode.WALK_IN,
     dict(heart_rate=82, resp_rate=15, systolic_bp=124, diastolic_bp=78, spo2=99, temp_c=36.8, blood_glucose=98, gcs_score=15, pain_score=5)),
    ("Sore throat and mild cough for a week", ArrivalMode.WALK_IN,
     dict(heart_rate=78, resp_rate=16, systolic_bp=118, diastolic_bp=76, spo2=98, temp_c=37.6, blood_glucose=102, gcs_score=15, pain_score=2)),
    ("Small cut on hand from kitchen knife", ArrivalMode.WALK_IN,
     dict(heart_rate=84, resp_rate=15, systolic_bp=122, diastolic_bp=80, spo2=99, temp_c=36.7, blood_glucose=95, gcs_score=15, pain_score=3)),
    ("Palpitations and dizziness since morning", ArrivalMode.WALK_IN,
     dict(heart_rate=146, resp_rate=20, systolic_bp=108, diastolic_bp=70, spo2=96, temp_c=36.9, blood_glucose=124, gcs_score=15, pain_score=2)),
]

WAITING = [
    ("Headache for three days", ArrivalMode.WALK_IN),
    ("Fall from motorbike, arm pain", ArrivalMode.AMBULANCE),
    ("Rash after new medication", ArrivalMode.WALK_IN),
]

APPOINTMENT_REASONS = [
    "Follow-up after discharge",
    "Blood pressure review",
    "Diabetes check-up",
    "Wound dressing change",
    "Asthma review",
    "Post-fever follow-up",
    "Chest pain follow-up, ECG results",
]


def _random_patient(rng: random.Random, i: int) -> Patient:
    female = rng.random() < 0.48
    first = rng.choice(FIRST_F if female else FIRST_M)
    age = rng.choice([rng.randint(19, 40), rng.randint(41, 64), rng.randint(65, 88)])
    dob = date.today().replace(year=date.today().year - age) - timedelta(days=rng.randint(0, 360))
    comorbidity = rng.choices(
        ["None", "Hypertension", "Diabetes", "Cardiovascular", "Asthma"], [45, 22, 15, 10, 8]
    )[0]
    return Patient(
        mrn=f"TC-{i:06d}",
        first_name=first,
        last_name=rng.choice(LAST),
        date_of_birth=dob,
        gender="Female" if female else "Male",
        phone=f"03{rng.randint(0, 4)}{rng.randint(0, 9)}-{rng.randint(1000000, 9999999)}",
        cnic=f"{rng.randint(35000, 45999)}-{rng.randint(1000000, 9999999)}-{rng.randint(1, 9)}",
        address=rng.choice(CITIES),
        comorbidity=comorbidity,
        allergies=rng.choice([None, None, None, "Penicillin", "NSAIDs", "Sulfa drugs"]),
        emergency_contact=f"{rng.choice(FIRST_M + FIRST_F)} (family) 03{rng.randint(0, 4)}{rng.randint(0, 9)}-{rng.randint(1000000, 9999999)}",
    )


def _backdate(visit: Visit, minutes_ago: int) -> None:
    arrived = utcnow() - timedelta(minutes=minutes_ago)
    visit.arrived_at = arrived
    if visit.triaged_at:
        visit.triaged_at = arrived + timedelta(minutes=4)
    for a in visit.assessments:
        a.created_at = arrived + timedelta(minutes=4)
    for n, e in enumerate(visit.events):
        e.created_at = arrived + timedelta(minutes=4 * n)


def seed(db: Session, password: str | None = None, rng_seed: int = 7) -> dict:
    if db.scalar(select(User).limit(1)):
        return {"skipped": True}

    rng = random.Random(rng_seed)
    password = password or get_settings().demo_password
    pw_hash = hash_password(password)

    for code, name, level, capacity, desc in WARDS:
        db.add(Ward(code=code, name=name, level=level, capacity=capacity, description=desc))
    staff = {}
    for username, name, role, specialty in STAFF:
        staff[username] = User(username=username, full_name=name, role=role, specialty=specialty, password_hash=pw_hash)
        db.add(staff[username])
    db.flush()

    patients = [_random_patient(rng, i) for i in range(1, 61)]
    db.add_all(patients)
    db.flush()

    nurse, reception = staff["nurse"], staff["reception"]
    pool = iter(rng.sample(patients, len(patients)))

    # Patients already triaged and waiting or being treated
    for n, (complaint, mode, vitals) in enumerate(SCENARIOS):
        patient = next(pool)
        visit = Visit(patient_id=patient.id, chief_complaint=complaint, arrival_mode=mode, registered_by_id=reception.id)
        db.add(visit)
        db.flush()
        log_event(db, visit, reception, "arrival", f"{'Arrived by ambulance' if mode == ArrivalMode.AMBULANCE else 'Walked in'}: {complaint}")
        db.flush()
        db.refresh(visit)
        record_triage(db, visit, vitals, nurse)
        db.flush()
        db.refresh(visit)
        minutes_ago = {"high": rng.randint(4, 12), "medium": rng.randint(10, 45)}.get(visit.tier, rng.randint(15, 110))
        if n in (0, 2, 6):
            visit.status = VisitStatus.IN_TREATMENT
            visit.seen_at = utcnow() - timedelta(minutes=max(1, minutes_ago - 20))
            log_event(db, visit, visit.doctor, "seen", f"Seen by {visit.doctor.display_name}")
            db.flush()
            db.refresh(visit)
        _backdate(visit, minutes_ago)
        if visit.seen_at:
            visit.seen_at = max(visit.seen_at, visit.triaged_at + timedelta(minutes=2))
            visit.events[-1].created_at = visit.seen_at

    # Just arrived, waiting for triage
    for complaint, mode in WAITING:
        patient = next(pool)
        visit = Visit(patient_id=patient.id, chief_complaint=complaint, arrival_mode=mode, registered_by_id=reception.id)
        db.add(visit)
        db.flush()
        log_event(db, visit, reception, "arrival", f"Arrived: {complaint}")
        db.flush()
        db.refresh(visit)
        _backdate(visit, rng.randint(1, 15))

    # Earlier visits that are finished, so patient histories aren't empty
    for _ in range(14):
        patient = next(pool)
        complaint, mode, vitals = rng.choice(SCENARIOS)
        visit = Visit(patient_id=patient.id, chief_complaint=complaint, arrival_mode=mode, registered_by_id=reception.id)
        db.add(visit)
        db.flush()
        db.refresh(visit)
        jittered = {k: (v + rng.uniform(-4, 4) if isinstance(v, int) and k not in ("gcs_score", "pain_score") else v)
                    for k, v in vitals.items()}
        record_triage(db, visit, jittered, nurse)
        days_ago = rng.randint(2, 90)
        visit.status = VisitStatus.COMPLETED
        visit.disposition = rng.choice(["discharged", "discharged", "discharged", "admitted", "referred"])
        db.flush()
        db.refresh(visit)
        _backdate(visit, days_ago * 24 * 60)
        visit.seen_at = visit.arrived_at + timedelta(minutes=rng.randint(10, 60))
        visit.completed_at = visit.arrived_at + timedelta(hours=rng.randint(1, 6))
        log_event(db, visit, visit.doctor, "complete", f"Visit closed: {visit.disposition}")
        db.flush()
        db.refresh(visit)
        visit.events[-1].created_at = visit.completed_at

    # Appointments today and tomorrow
    doctors = [u for u in staff.values() if u.role == Role.DOCTOR]
    today = local_today()
    for day_offset, count in ((0, 8), (1, 6)):
        for slot in range(count):
            patient = next(pool)
            local_time = datetime.combine(today + timedelta(days=day_offset), time(9, 0)) + timedelta(minutes=40 * slot)
            appt = Appointment(
                patient_id=patient.id,
                doctor_id=rng.choice(doctors).id,
                scheduled_at=to_utc(local_time),
                reason=rng.choice(APPOINTMENT_REASONS),
                created_by_id=reception.id,
            )
            if day_offset == 0 and slot == 0:
                appt.status = AppointmentStatus.NO_SHOW
            db.add(appt)

    db.commit()
    return {"skipped": False, "users": len(STAFF), "patients": len(patients)}
