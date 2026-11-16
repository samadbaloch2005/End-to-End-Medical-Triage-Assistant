from fastapi import APIRouter, Depends, Form, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from triagecare.db import get_db
from triagecare.deps import current_user, current_user_optional, verify_csrf
from triagecare.models import Role, User
from triagecare.security import verify_password
from triagecare.web import flash, redirect, render

router = APIRouter(dependencies=[Depends(verify_csrf)])


def _safe_next(url: str | None) -> str:
    # Only allow local paths, so the login form can't be used as an open redirect.
    if url and url.startswith("/") and not url.startswith("//"):
        return url
    return "/"


@router.get("/login")
def login_page(request: Request, next: str | None = None, user: User | None = Depends(current_user_optional)):
    if user:
        return redirect(_safe_next(next))
    return render(request, "auth/login.html", {"next": next})


@router.post("/login")
def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next: str | None = Form(None),
    db: Session = Depends(get_db),
):
    user = db.scalar(select(User).where(User.username == username.strip().lower()))
    if user is None or not user.is_active or not verify_password(password, user.password_hash):
        return render(
            request,
            "auth/login.html",
            {"next": next, "error": "That username and password don't match.", "username": username},
            status_code=400,
        )
    request.session.clear()
    request.session["user_id"] = user.id
    return redirect(_safe_next(next))


@router.post("/logout")
def logout(request: Request):
    request.session.clear()
    return redirect("/login")


@router.post("/me/duty")
def toggle_duty(request: Request, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if user.role != Role.DOCTOR:
        return redirect("/", request)
    user.on_duty = not user.on_duty
    db.commit()
    flash(request, "You're on duty and can receive patients." if user.on_duty else "You're off duty. New patients won't be assigned to you.")
    return redirect(request.headers.get("referer") or "/", request)
