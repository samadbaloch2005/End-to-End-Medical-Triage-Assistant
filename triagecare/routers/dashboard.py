from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from triagecare.db import get_db
from triagecare.deps import current_user
from triagecare.models import Role, User, VisitEvent
from triagecare.services.queue import build_queue, dashboard_stats, todays_appointments, ward_board
from triagecare.timeutil import to_local, utcnow
from triagecare.web import render

router = APIRouter()


@router.get("/")
def dashboard(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    recent = db.scalars(
        select(VisitEvent)
        .options(selectinload(VisitEvent.visit), selectinload(VisitEvent.user))
        .order_by(VisitEvent.created_at.desc())
        .limit(8)
    ).all()
    queue = build_queue(db)
    my_queue = build_queue(db, doctor_id=user.id) if user.role == Role.DOCTOR else None
    now = to_local(utcnow())
    greeting = "morning" if now.hour < 12 else "afternoon" if now.hour < 17 else "evening"
    return render(
        request,
        "dashboard.html",
        {
            "user": user,
            "greeting": greeting,
            "now_label": now.strftime("%A, %d %B"),
            "stats": dashboard_stats(db),
            "queue": queue,
            "my_queue": my_queue,
            "wards": ward_board(db),
            "appointments": todays_appointments(db)[:6],
            "recent": recent,
        },
    )
