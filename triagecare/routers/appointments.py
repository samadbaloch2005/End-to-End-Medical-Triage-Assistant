from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from triagecare.db import get_db
from triagecare.deps import current_user, require_roles, verify_csrf
from triagecare.models import (
    Appointment,
    AppointmentStatus,
    ArrivalMode,
    Patient,
    Role,
    User,
    Visit,
)
from triagecare.schemas import AppointmentIn, parse_form
from triagecare.services.queue import todays_appointments
from triagecare.services.triage import log_event
from triagecare.timeutil import local_today, to_utc, utcnow
from triagecare.web import flash, redirect, render

router = APIRouter(prefix="/appointments", dependencies=[Depends(verify_csrf)])

front_desk = require_roles(Role.RECEPTIONIST, Role.NURSE)


def _doctors(db: Session) -> list[User]:
    return list(
        db.scalars(select(User).where(User.role == Role.DOCTOR, User.is_active).order_by(User.full_name))
    )


@router.get("")
def list_appointments(
    request: Request,
    day: date | None = None,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    day = day or local_today()
    appointments = todays_appointments(db, day)
    if user.role == Role.DOCTOR and request.query_params.get("mine"):
        appointments = [a for a in appointments if a.doctor_id == user.id]
    return render(
        request,
        "appointments/list.html",
        {
            "user": user,
            "day": day,
            "prev_day": day - timedelta(days=1),
            "next_day": day + timedelta(days=1),
            "is_today": day == local_today(),
            "appointments": appointments,
            "counts": {
                s.value: sum(1 for a in appointments if a.status == s) for s in AppointmentStatus
            },
        },
    )


@router.get("/new")
def new_appointment(
    request: Request,
    patient_id: int | None = None,
    user: User = Depends(front_desk),
    db: Session = Depends(get_db),
):
    patient = db.get(Patient, patient_id) if patient_id else None
    return render(
        request,
        "appointments/form.html",
        {"user": user, "patient": patient, "doctors": _doctors(db), "values": {}, "errors": {}},
    )


@router.post("/new")
async def create_appointment(request: Request, user: User = Depends(front_desk), db: Session = Depends(get_db)):
    form = await request.form()
    data, errors = parse_form(AppointmentIn, form)
    patient = db.get(Patient, data.patient_id) if data else None
    if data and patient is None:
        errors = {"patient_id": "Choose a patient"}
    if data and not errors:
        when = to_utc(data.scheduled_at)
        if when < utcnow() - timedelta(minutes=5):
            errors = {"scheduled_at": "That time has already passed"}
    if errors:
        pid = form.get("patient_id")
        return render(
            request,
            "appointments/form.html",
            {"user": user, "patient": db.get(Patient, int(pid)) if pid and pid.isdigit() else None,
             "doctors": _doctors(db), "values": form, "errors": errors},
            status_code=422,
        )
    appt = Appointment(
        patient_id=patient.id,
        doctor_id=data.doctor_id,
        scheduled_at=when,
        reason=data.reason,
        created_by_id=user.id,
    )
    db.add(appt)
    db.commit()
    flash(request, f"Appointment booked for {patient.full_name}.")
    local_day = data.scheduled_at.date().isoformat()
    return redirect(f"/appointments?day={local_day}")


def _get(db: Session, appointment_id: int) -> Appointment:
    appt = db.get(Appointment, appointment_id)
    if appt is None:
        raise HTTPException(404, "Appointment not found")
    return appt


@router.post("/{appointment_id}/check-in")
def check_in(request: Request, appointment_id: int, user: User = Depends(front_desk), db: Session = Depends(get_db)):
    appt = _get(db, appointment_id)
    if appt.status != AppointmentStatus.BOOKED:
        flash(request, "This appointment can't be checked in.", "error")
        return redirect("/appointments", request)
    existing = db.scalar(
        select(Visit).where(Visit.patient_id == appt.patient_id, Visit.status != "completed")
    )
    if existing:
        flash(request, f"{appt.patient.full_name} already has an open visit.", "error")
        return redirect(f"/visits/{existing.id}", request)

    visit = Visit(
        patient_id=appt.patient_id,
        appointment_id=appt.id,
        arrival_mode=ArrivalMode.APPOINTMENT,
        chief_complaint=appt.reason,
        registered_by_id=user.id,
    )
    appt.status = AppointmentStatus.CHECKED_IN
    db.add(visit)
    db.flush()
    log_event(db, visit, user, "arrival", "Checked in for appointment")
    db.commit()
    flash(request, f"{appt.patient.full_name} checked in and added to the triage queue.")
    return redirect(request.headers.get("referer") or "/appointments", request)


@router.post("/{appointment_id}/status")
async def set_status(request: Request, appointment_id: int, user: User = Depends(front_desk), db: Session = Depends(get_db)):
    form = await request.form()
    new_status = form.get("status")
    if new_status not in (AppointmentStatus.CANCELLED, AppointmentStatus.NO_SHOW, AppointmentStatus.BOOKED):
        raise HTTPException(400, "Unknown status")
    appt = _get(db, appointment_id)
    if appt.status == AppointmentStatus.CHECKED_IN:
        flash(request, "Checked-in appointments can't be changed.", "error")
    else:
        appt.status = new_status
        db.commit()
        flash(request, f"Appointment marked as {new_status.replace('_', ' ')}.")
    return redirect(request.headers.get("referer") or "/appointments", request)
