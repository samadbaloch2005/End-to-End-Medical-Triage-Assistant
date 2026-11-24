from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from triagecare.db import get_db
from triagecare.deps import current_user, require_roles, verify_csrf
from triagecare.models import (
    ArrivalMode,
    Patient,
    Role,
    Tier,
    TriageAssessment,
    User,
    Visit,
    VisitEvent,
    VisitStatus,
    Ward,
)
from triagecare.schemas import CompleteIn, OverrideIn, VisitIn, VitalsIn, parse_form
from triagecare.services.triage import ModelUnavailable, assess, log_event, override_tier, record_triage
from triagecare.timeutil import utcnow
from triagecare.web import flash, redirect, render

router = APIRouter(prefix="/visits", dependencies=[Depends(verify_csrf)])

registers = require_roles(Role.RECEPTIONIST, Role.NURSE)
clinicians = require_roles(Role.NURSE, Role.DOCTOR)
doctors = require_roles(Role.DOCTOR)


def _get_visit(db: Session, visit_id: int) -> Visit:
    visit = db.get(
        Visit,
        visit_id,
        options=[
            selectinload(Visit.patient),
            selectinload(Visit.ward),
            selectinload(Visit.doctor),
            selectinload(Visit.assessments).selectinload(TriageAssessment.nurse),
            selectinload(Visit.assessments).selectinload(TriageAssessment.recommended_ward),
            selectinload(Visit.assessments).selectinload(TriageAssessment.recommended_doctor),
            selectinload(Visit.events).selectinload(VisitEvent.user),
        ],
    )
    if visit is None:
        raise HTTPException(404, "Visit not found")
    return visit


# ---------------------------------------------------------------- arrival
@router.get("/new")
def new_visit(request: Request, patient_id: int | None = None, user: User = Depends(registers), db: Session = Depends(get_db)):
    patient = db.get(Patient, patient_id) if patient_id else None
    open_visit = None
    if patient:
        open_visit = db.scalar(
            select(Visit).where(Visit.patient_id == patient.id, Visit.status != VisitStatus.COMPLETED)
        )
    return render(
        request,
        "visits/new.html",
        {"user": user, "patient": patient, "open_visit": open_visit, "values": {}, "errors": {},
         "modes": [ArrivalMode.WALK_IN, ArrivalMode.AMBULANCE]},
    )


@router.post("/new")
async def create_visit(request: Request, user: User = Depends(registers), db: Session = Depends(get_db)):
    form = await request.form()
    data, errors = parse_form(VisitIn, form)
    patient = db.get(Patient, data.patient_id) if data else None
    if data and patient is None:
        errors = {"patient_id": "Choose a patient"}
    if errors:
        pid = form.get("patient_id")
        return render(
            request,
            "visits/new.html",
            {"user": user, "patient": db.get(Patient, int(pid)) if pid and pid.isdigit() else None,
             "values": form, "errors": errors, "modes": [ArrivalMode.WALK_IN, ArrivalMode.AMBULANCE]},
            status_code=422,
        )
    existing = db.scalar(
        select(Visit).where(Visit.patient_id == patient.id, Visit.status != VisitStatus.COMPLETED)
    )
    if existing:
        flash(request, f"{patient.full_name} is already in the department.", "error")
        return redirect(f"/visits/{existing.id}")

    visit = Visit(
        patient_id=patient.id,
        chief_complaint=data.chief_complaint,
        arrival_mode=data.arrival_mode,
        registered_by_id=user.id,
    )
    db.add(visit)
    db.flush()
    how = "Arrived by ambulance" if data.arrival_mode == ArrivalMode.AMBULANCE else "Walked in"
    log_event(db, visit, user, "arrival", f"{how}: {data.chief_complaint}")
    db.commit()
    flash(request, f"{patient.full_name} added to the triage queue.")
    if user.role == Role.NURSE:
        return redirect(f"/visits/{visit.id}/triage")
    return redirect("/queue")


# ---------------------------------------------------------------- detail
@router.get("/{visit_id}")
def visit_detail(request: Request, visit_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    visit = _get_visit(db, visit_id)
    return render(
        request,
        "visits/detail.html",
        {
            "user": user,
            "visit": visit,
            "a": visit.latest_assessment,
            "wards": db.scalars(select(Ward)).all(),
            "doctor_list": db.scalars(
                select(User).where(User.role == Role.DOCTOR, User.is_active).order_by(User.full_name)
            ).all(),
        },
    )


# ---------------------------------------------------------------- triage
@router.get("/{visit_id}/triage")
def triage_form(request: Request, visit_id: int, user: User = Depends(clinicians), db: Session = Depends(get_db)):
    visit = _get_visit(db, visit_id)
    if not visit.is_active:
        flash(request, "This visit is closed.", "error")
        return redirect(f"/visits/{visit.id}")
    last = visit.latest_assessment
    values = {}
    if last:
        values = {f: getattr(last, f) for f in VitalsIn.model_fields if f != "notes"}
    return render(request, "visits/triage.html", {"user": user, "visit": visit, "values": values, "errors": {}})


@router.post("/{visit_id}/triage/preview")
async def triage_preview(request: Request, visit_id: int, user: User = Depends(clinicians), db: Session = Depends(get_db)):
    """Live risk preview while the nurse is typing. Nothing is saved."""
    visit = _get_visit(db, visit_id)
    form = await request.form()
    data, errors = parse_form(VitalsIn, form)
    if errors:
        return render(request, "visits/_preview.html", {"incomplete": True, "errors": errors})
    try:
        result = assess(db, data.model_dump(), visit.patient.age, visit.patient.gender, visit.patient.comorbidity, visit)
    except ModelUnavailable as exc:
        return render(request, "visits/_preview.html", {"model_error": str(exc)})
    return render(request, "visits/_preview.html", {"r": result, "vitals": data.model_dump()})


@router.post("/{visit_id}/triage")
async def submit_triage(request: Request, visit_id: int, user: User = Depends(clinicians), db: Session = Depends(get_db)):
    visit = _get_visit(db, visit_id)
    form = await request.form()
    data, errors = parse_form(VitalsIn, form)
    if errors:
        return render(
            request, "visits/triage.html", {"user": user, "visit": visit, "values": form, "errors": errors},
            status_code=422,
        )
    try:
        assessment = record_triage(db, visit, data.model_dump(exclude={"notes"}), user, data.notes)
    except ModelUnavailable as exc:
        flash(request, str(exc), "error")
        return redirect(f"/visits/{visit.id}/triage")
    db.commit()
    flash(request, f"Triage saved: {assessment.final_tier.title()} risk.")
    return redirect(f"/visits/{visit.id}")


# ---------------------------------------------------------------- care
@router.post("/{visit_id}/start")
def start_treatment(request: Request, visit_id: int, user: User = Depends(doctors), db: Session = Depends(get_db)):
    visit = _get_visit(db, visit_id)
    if visit.status != VisitStatus.TRIAGED:
        flash(request, "Only triaged patients can be picked up.", "error")
        return redirect(f"/visits/{visit.id}", request)
    visit.status = VisitStatus.IN_TREATMENT
    visit.seen_at = utcnow()
    if user.role == Role.DOCTOR:
        visit.doctor_id = user.id
    log_event(db, visit, user, "seen", f"Seen by {user.display_name}")
    db.commit()
    flash(request, f"You are now treating {visit.patient.full_name}.")
    return redirect(f"/visits/{visit.id}", request)


@router.post("/{visit_id}/assign")
async def assign(request: Request, visit_id: int, user: User = Depends(clinicians), db: Session = Depends(get_db)):
    visit = _get_visit(db, visit_id)
    form = await request.form()
    changes = []
    ward_id = form.get("ward_id")
    if ward_id and ward_id.isdigit() and int(ward_id) != visit.ward_id:
        ward = db.get(Ward, int(ward_id))
        if ward:
            visit.ward_id = ward.id
            changes.append(f"ward to {ward.name}")
    doctor_id = form.get("doctor_id")
    if doctor_id and doctor_id.isdigit() and int(doctor_id) != visit.doctor_id:
        doctor = db.get(User, int(doctor_id))
        if doctor and doctor.role == Role.DOCTOR:
            visit.doctor_id = doctor.id
            changes.append(f"doctor to {doctor.display_name}")
    if changes:
        log_event(db, visit, user, "assign", "Changed " + " and ".join(changes))
        db.commit()
        flash(request, "Assignment updated.")
    return redirect(f"/visits/{visit.id}", request)


@router.post("/{visit_id}/override")
async def override(request: Request, visit_id: int, user: User = Depends(clinicians), db: Session = Depends(get_db)):
    visit = _get_visit(db, visit_id)
    form = await request.form()
    data, errors = parse_form(OverrideIn, form)
    if errors:
        flash(request, "Give a priority and a short reason for the change.", "error")
        return redirect(f"/visits/{visit.id}", request)
    try:
        override_tier(db, visit, Tier(data.tier), data.reason, user)
    except ValueError as exc:
        flash(request, str(exc), "error")
        return redirect(f"/visits/{visit.id}", request)
    db.commit()
    flash(request, f"Priority changed to {data.tier.value.title()}.")
    return redirect(f"/visits/{visit.id}", request)


@router.post("/{visit_id}/complete")
async def complete(request: Request, visit_id: int, user: User = Depends(clinicians), db: Session = Depends(get_db)):
    visit = _get_visit(db, visit_id)
    form = await request.form()
    data, errors = parse_form(CompleteIn, form)
    if errors:
        flash(request, "Choose an outcome for the visit.", "error")
        return redirect(f"/visits/{visit.id}", request)
    if not visit.is_active:
        return redirect(f"/visits/{visit.id}", request)
    visit.status = VisitStatus.COMPLETED
    visit.completed_at = utcnow()
    visit.disposition = data.disposition
    visit.doctor_notes = data.doctor_notes
    log_event(db, visit, user, "complete", f"Visit closed: {data.disposition.replace('_', ' ')}")
    db.commit()
    flash(request, f"Visit closed for {visit.patient.full_name}.")
    return redirect("/queue", request)
