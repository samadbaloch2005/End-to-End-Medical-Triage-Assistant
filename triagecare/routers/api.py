"""JSON API. Uses the same session cookie as the web pages, so it can be tried
from the interactive docs at /docs once you are signed in."""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from triagecare.db import get_db
from triagecare.deps import current_user, require_roles, verify_csrf
from triagecare.models import Patient, Role, User
from triagecare.routers.patients import search_patients
from triagecare.schemas import PredictIn
from triagecare.services.queue import build_queue, ward_board
from triagecare.services.triage import ModelUnavailable, assess, get_model

router = APIRouter(prefix="/api/v1", tags=["api"], dependencies=[Depends(verify_csrf)])


class PatientOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    mrn: str
    full_name: str
    age: int
    gender: str
    phone: str | None
    comorbidity: str


class QueueItemOut(BaseModel):
    visit_id: int
    position: int
    patient: PatientOut
    status: str
    tier: str | None
    chief_complaint: str
    arrived_at: datetime
    waited_minutes: int
    target_minutes: int | None
    overdue: bool
    ward: str | None
    doctor: str | None


class FlagOut(BaseModel):
    level: str
    vital: str
    message: str


class DriverOut(BaseModel):
    feature: str
    value: float | int | str
    impact: float


class PredictionOut(BaseModel):
    tier: str
    model_tier: str
    probabilities: dict[str, float]
    red_flags: list[FlagOut]
    drivers: list[DriverOut]
    specialty: str
    recommended_ward: str | None
    recommended_doctor: str | None
    model_version: str


@router.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "model_loaded": get_model().available}


@router.get("/me")
def me(user: User = Depends(current_user)):
    return {"id": user.id, "username": user.username, "name": user.full_name, "role": user.role}


@router.post("/predict", response_model=PredictionOut)
def predict(body: PredictIn, user: User = Depends(require_roles(Role.NURSE, Role.DOCTOR)), db: Session = Depends(get_db)):
    """Score a set of vitals without creating a patient or visit."""
    vitals = body.model_dump(exclude={"age", "gender", "comorbidity", "notes"})
    try:
        r = assess(db, vitals, body.age, body.gender, body.comorbidity)
    except ModelUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return PredictionOut(
        tier=r.final_tier,
        model_tier=r.model_tier,
        probabilities=r.probabilities,
        red_flags=[f.as_dict() for f in r.flags],
        drivers=r.drivers,
        specialty=r.specialty,
        recommended_ward=r.ward.name if r.ward else None,
        recommended_doctor=r.doctor.display_name if r.doctor else None,
        model_version=r.model_version,
    )


@router.get("/queue", response_model=list[QueueItemOut])
def queue(user: User = Depends(current_user), db: Session = Depends(get_db)):
    lanes = build_queue(db)
    items = []
    for lane in ("awaiting", "triaged", "treating"):
        for e in lanes[lane]:
            v = e.visit
            items.append(
                QueueItemOut(
                    visit_id=v.id,
                    position=e.position,
                    patient=PatientOut.model_validate(v.patient),
                    status=v.status,
                    tier=v.tier,
                    chief_complaint=v.chief_complaint,
                    arrived_at=v.arrived_at,
                    waited_minutes=e.waited,
                    target_minutes=e.target,
                    overdue=e.overdue,
                    ward=v.ward.name if v.ward else None,
                    doctor=v.doctor.display_name if v.doctor else None,
                )
            )
    return items


@router.get("/wards")
def wards(user: User = Depends(current_user), db: Session = Depends(get_db)):
    return [
        {"code": w["ward"].code, "name": w["ward"].name, "level": w["ward"].level,
         "capacity": w["ward"].capacity, "occupied": w["occupied"], "free": w["free"]}
        for w in ward_board(db)
    ]


@router.get("/patients", response_model=list[PatientOut])
def patients(q: str | None = None, limit: int = 25, user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows, _ = search_patients(db, q, limit=min(limit, 100))
    return rows


@router.get("/patients/{patient_id}", response_model=PatientOut)
def patient(patient_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    p = db.get(Patient, patient_id)
    if p is None:
        raise HTTPException(404, "Patient not found")
    return p
