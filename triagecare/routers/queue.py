from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from triagecare.db import get_db
from triagecare.deps import current_user
from triagecare.models import Role, User
from triagecare.services.queue import build_queue, ward_board
from triagecare.web import render

router = APIRouter()


@router.get("/queue")
def queue_page(request: Request, mine: bool = False, user: User = Depends(current_user), db: Session = Depends(get_db)):
    mine = mine and user.role == Role.DOCTOR
    ctx = {
        "user": user,
        "queue": build_queue(db, doctor_id=user.id if mine else None),
        "wards": ward_board(db),
        "mine": mine,
    }
    # The board polls itself every few seconds; only the inner part is re-rendered.
    if request.headers.get("hx-target") == "queue-board":
        return render(request, "queue/_board.html", ctx)
    return render(request, "queue/index.html", ctx)


@router.get("/wards")
def wards_page(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    queue = build_queue(db)
    by_ward: dict[int, list] = {}
    for lane in ("triaged", "treating"):
        for entry in queue[lane]:
            if entry.visit.ward_id:
                by_ward.setdefault(entry.visit.ward_id, []).append(entry)
    return render(request, "wards.html", {"user": user, "wards": ward_board(db), "by_ward": by_ward})
