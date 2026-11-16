from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, selectinload

from triagecare.db import get_db
from triagecare.deps import current_user, require_roles, verify_csrf
from triagecare.models import Patient, Role, TriageAssessment, User, Visit
from triagecare.schemas import PatientIn, parse_form
from triagecare.services.triage import COMORBIDITIES
from triagecare.web import flash, redirect, render

router = APIRouter(prefix="/patients", dependencies=[Depends(verify_csrf)])

PAGE_SIZE = 25


def search_patients(db: Session, q: str | None, limit: int = PAGE_SIZE, offset: int = 0):
    stmt = select(Patient)
    if q:
        term = f"%{q.strip().lower()}%"
        full_name = func.lower(Patient.first_name + " " + Patient.last_name)
        stmt = stmt.where(
            or_(
                full_name.like(term),
                func.lower(Patient.mrn).like(term),
                Patient.phone.like(term),
                Patient.cnic.like(term),
            )
        )
    total = db.scalar(select(func.count()).select_from(stmt.subquery()))
    rows = db.scalars(stmt.order_by(Patient.created_at.desc()).limit(limit).offset(offset)).all()
    return rows, total


def next_mrn(db: Session) -> str:
    last = db.scalar(select(func.max(Patient.id))) or 0
    return f"TC-{last + 1:06d}"


@router.get("")
def list_patients(
    request: Request,
    q: str | None = None,
    page: int = 1,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    page = max(1, page)
    patients, total = search_patients(db, q, offset=(page - 1) * PAGE_SIZE)
    ctx = {
        "user": user,
        "patients": patients,
        "q": q or "",
        "total": total,
        "page": page,
        "pages": max(1, -(-total // PAGE_SIZE)),
        "pick": request.query_params.get("pick"),
    }
    if request.headers.get("hx-target") == "patient-results":
        return render(request, "patients/_results.html", ctx)
    return render(request, "patients/list.html", ctx)


@router.get("/new")
def new_patient(request: Request, user: User = Depends(require_roles(Role.RECEPTIONIST, Role.NURSE))):
    return render(
        request,
        "patients/form.html",
        {"user": user, "patient": None, "values": {}, "errors": {}, "comorbidities": COMORBIDITIES,
         "then": request.query_params.get("then")},
    )


@router.post("/new")
async def create_patient(
    request: Request,
    user: User = Depends(require_roles(Role.RECEPTIONIST, Role.NURSE)),
    db: Session = Depends(get_db),
):
    form = await request.form()
    data, errors = parse_form(PatientIn, form)
    if errors:
        return render(
            request,
            "patients/form.html",
            {"user": user, "patient": None, "values": form, "errors": errors,
             "comorbidities": COMORBIDITIES, "then": form.get("then")},
            status_code=422,
        )
    patient = Patient(mrn=next_mrn(db), **data.model_dump())
    db.add(patient)
    db.commit()
    flash(request, f"{patient.full_name} registered as {patient.mrn}.")

    then = form.get("then")
    if then == "visit":
        return redirect(f"/visits/new?patient_id={patient.id}")
    if then == "appointment":
        return redirect(f"/appointments/new?patient_id={patient.id}")
    return redirect(f"/patients/{patient.id}")


def _get_patient(db: Session, patient_id: int) -> Patient:
    patient = db.get(
        Patient,
        patient_id,
        options=[
            selectinload(Patient.visits).selectinload(Visit.assessments),
            selectinload(Patient.visits).selectinload(Visit.ward),
            selectinload(Patient.visits).selectinload(Visit.doctor),
            selectinload(Patient.appointments),
        ],
    )
    if patient is None:
        raise HTTPException(404, "Patient not found")
    return patient


@router.get("/{patient_id}")
def patient_detail(request: Request, patient_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    patient = _get_patient(db, patient_id)
    assessments = db.scalars(
        select(TriageAssessment)
        .join(Visit)
        .where(Visit.patient_id == patient.id)
        .order_by(TriageAssessment.created_at)
    ).all()
    trend = {
        "labels": [a.created_at.isoformat() + "Z" for a in assessments],
        "heart_rate": [a.heart_rate for a in assessments],
        "spo2": [a.spo2 for a in assessments],
        "systolic_bp": [a.systolic_bp for a in assessments],
        "resp_rate": [a.resp_rate for a in assessments],
    }
    active = next((v for v in patient.visits if v.is_active), None)
    return render(
        request,
        "patients/detail.html",
        {"user": user, "patient": patient, "trend": trend, "active_visit": active,
         "assessment_count": len(assessments)},
    )


@router.get("/{patient_id}/edit")
def edit_patient(
    request: Request,
    patient_id: int,
    user: User = Depends(require_roles(Role.RECEPTIONIST, Role.NURSE)),
    db: Session = Depends(get_db),
):
    patient = _get_patient(db, patient_id)
    values = {c: getattr(patient, c) for c in PatientIn.model_fields}
    return render(
        request,
        "patients/form.html",
        {"user": user, "patient": patient, "values": values, "errors": {}, "comorbidities": COMORBIDITIES},
    )


@router.post("/{patient_id}/edit")
async def update_patient(
    request: Request,
    patient_id: int,
    user: User = Depends(require_roles(Role.RECEPTIONIST, Role.NURSE)),
    db: Session = Depends(get_db),
):
    patient = _get_patient(db, patient_id)
    form = await request.form()
    data, errors = parse_form(PatientIn, form)
    if errors:
        return render(
            request,
            "patients/form.html",
            {"user": user, "patient": patient, "values": form, "errors": errors, "comorbidities": COMORBIDITIES},
            status_code=422,
        )
    for key, value in data.model_dump().items():
        setattr(patient, key, value)
    db.commit()
    flash(request, "Patient details updated.")
    return redirect(f"/patients/{patient.id}")
