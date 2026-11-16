from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from triagecare.db import get_db
from triagecare.deps import require_roles, verify_csrf
from triagecare.models import Role, TriageAssessment, User
from triagecare.schemas import UserIn, parse_form
from triagecare.security import hash_password
from triagecare.services.triage import SPECIALTIES, get_model
from triagecare.web import flash, redirect, render

router = APIRouter(prefix="/admin", dependencies=[Depends(verify_csrf)])
admin_only = require_roles()  # admins pass every role check
ROLE_OPTIONS = [(r.value, r.value.title()) for r in Role]


@router.get("/users")
def users(request: Request, user: User = Depends(admin_only), db: Session = Depends(get_db)):
    staff = db.scalars(select(User).order_by(User.role, User.full_name)).all()
    return render(
        request,
        "admin/users.html",
        {"user": user, "staff": staff, "roles": ROLE_OPTIONS, "specialties": SPECIALTIES, "values": {}, "errors": {}},
    )


@router.post("/users")
async def create_user(request: Request, user: User = Depends(admin_only), db: Session = Depends(get_db)):
    form = await request.form()
    data, errors = parse_form(UserIn, form)
    if data and db.scalar(select(User).where(User.username == data.username)):
        errors = {"username": "Username is taken"}
    if data and data.role == Role.DOCTOR and not data.specialty:
        errors = {"specialty": "Doctors need a specialty"}
    if errors:
        staff = db.scalars(select(User).order_by(User.role, User.full_name)).all()
        return render(
            request,
            "admin/users.html",
            {"user": user, "staff": staff, "roles": ROLE_OPTIONS, "specialties": SPECIALTIES,
             "values": form, "errors": errors},
            status_code=422,
        )
    db.add(
        User(
            username=data.username,
            full_name=data.full_name,
            role=data.role,
            specialty=data.specialty if data.role == Role.DOCTOR else None,
            password_hash=hash_password(data.password),
        )
    )
    db.commit()
    flash(request, f"Account created for {data.full_name}.")
    return redirect("/admin/users")


@router.post("/users/{user_id}/active")
def toggle_active(request: Request, user_id: int, user: User = Depends(admin_only), db: Session = Depends(get_db)):
    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(404)
    if target.id == user.id:
        flash(request, "You can't deactivate your own account.", "error")
    else:
        target.is_active = not target.is_active
        db.commit()
        flash(request, f"{target.full_name} {'reactivated' if target.is_active else 'deactivated'}.")
    return redirect("/admin/users", request)


@router.get("/model")
def model_page(request: Request, user: User = Depends(admin_only), db: Session = Depends(get_db)):
    model = get_model()
    card = model.card if model.available else None

    total = db.scalar(select(func.count(TriageAssessment.id))) or 0
    overridden = db.scalar(
        select(func.count(TriageAssessment.id)).where(TriageAssessment.override_tier.is_not(None))
    ) or 0
    raised = db.scalar(
        select(func.count(TriageAssessment.id)).where(TriageAssessment.final_tier != TriageAssessment.model_tier)
    ) or 0
    live = {
        "assessments": total,
        "overridden": overridden,
        "raised_by_rules": raised,
        "override_rate": overridden / total if total else 0,
    }
    return render(request, "admin/model.html", {"user": user, "card": card, "live": live})
