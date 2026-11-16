"""Request dependencies: database session, signed-in user, role checks, CSRF."""
import hmac
from collections.abc import Callable

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from triagecare.db import get_db
from triagecare.models import Role, User
from triagecare.security import new_csrf_token


class LoginRequired(Exception):
    """Raised for page requests without a session; handled by redirecting to /login."""


def csrf_token(request: Request) -> str:
    token = request.session.get("csrf")
    if not token:
        token = new_csrf_token()
        request.session["csrf"] = token
    return token


async def verify_csrf(request: Request) -> None:
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    # Browsers can't send a cross-site JSON body without a CORS preflight, which
    # this app never approves, so JSON API calls are already protected.
    if request.headers.get("content-type", "").startswith("application/json"):
        return
    expected = request.session.get("csrf")
    sent = request.headers.get("x-csrf-token")
    if sent is None and request.headers.get("content-type", "").startswith(
        ("application/x-www-form-urlencoded", "multipart/form-data")
    ):
        form = await request.form()
        sent = form.get("csrf_token")
    if not expected or not sent or not hmac.compare_digest(str(sent), expected):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid or missing CSRF token")


def current_user_optional(request: Request, db: Session = Depends(get_db)) -> User | None:
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        request.session.clear()
        return None
    return user


def current_user(request: Request, user: User | None = Depends(current_user_optional)) -> User:
    if user is None:
        if request.url.path.startswith("/api/"):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not signed in")
        raise LoginRequired()
    return user


def require_roles(*roles: Role) -> Callable[..., User]:
    """Allow only the given roles. Admins are always allowed."""

    def checker(user: User = Depends(current_user)) -> User:
        if user.role != Role.ADMIN and user.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Your role can't do that")
        return user

    return checker
