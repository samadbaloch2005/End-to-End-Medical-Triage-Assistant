"""Template rendering, filters and flash messages."""
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from triagecare import __version__
from triagecare.config import get_settings
from triagecare.deps import csrf_token
from triagecare.services.clinical import care_plan, possible_concerns, read_vitals, vitals_from_assessment
from triagecare.services.triage import FEATURE_LABELS, TARGET_MINUTES, describe_driver
from triagecare.timeutil import humanize_minutes, minutes_since, to_local

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def _fmt(dt: datetime | None, pattern: str) -> str:
    local = to_local(dt)
    return local.strftime(pattern) if local else "—"


def _label(value: str | None) -> str:
    if not value:
        return "—"
    return value.replace("_", " ").capitalize()


def _ago(dt: datetime | None) -> str:
    if dt is None:
        return "\u2014"
    text = humanize_minutes(minutes_since(dt))
    return text if text == "just now" else f"{text} ago"


env = templates.env
env.filters["time"] = lambda dt: _fmt(dt, "%H:%M")
env.filters["date"] = lambda dt: _fmt(dt, "%d %b %Y")
env.filters["datetime"] = lambda dt: _fmt(dt, "%d %b %Y, %H:%M")
env.filters["ago"] = _ago
env.filters["minutes"] = humanize_minutes
env.filters["label"] = _label
env.filters["pct"] = lambda x: f"{x * 100:.0f}%"
env.filters["feature"] = lambda f: FEATURE_LABELS.get(f, f)
env.filters["driver_value"] = describe_driver
env.filters["doctor_options"] = lambda ds: [(d.id, f"{d.display_name} \u00b7 {d.specialty}") for d in ds]
env.filters["zip_pairs"] = lambda items: [(i, i) for i in items]
env.globals["TARGET_MINUTES"] = TARGET_MINUTES
env.globals["app_version"] = __version__
env.globals["read_vitals"] = read_vitals
env.globals["possible_concerns"] = possible_concerns
env.globals["care_plan"] = care_plan
env.globals["vitals_of"] = vitals_from_assessment


def flash(request: Request, message: str, kind: str = "success") -> None:
    request.session.setdefault("flashes", []).append({"kind": kind, "message": message})


def _pop_flashes(request: Request) -> list[dict]:
    return request.session.pop("flashes", [])


def side_queue() -> dict:
    """Patients waiting for triage or a doctor, for the live list in the sidebar."""
    from triagecare.db import SessionLocal
    from triagecare.services.queue import build_queue

    with SessionLocal() as db:
        q = build_queue(db)
        items = [
            {"id": e.visit.id, "name": e.visit.patient.full_name, "tier": e.visit.tier or "waiting",
             "waited": e.waited, "overdue": e.overdue, "status": e.visit.status}
            for e in q["triaged"]
        ] + [
            {"id": e.visit.id, "name": e.visit.patient.full_name, "tier": "waiting",
             "waited": e.waited, "overdue": False, "status": e.visit.status}
            for e in q["awaiting"]
        ]
    return {"count": len(items), "items": items}


def render(request: Request, name: str, context: dict[str, Any] | None = None, status_code: int = 200) -> HTMLResponse:
    settings = get_settings()
    ctx = {
        "settings": settings,
        "csrf": csrf_token(request),
        "flashes": _pop_flashes(request),
        "path": request.url.path,
        "is_htmx": request.headers.get("hx-request") == "true",
    }
    ctx.update(context or {})
    if ctx.get("user") and not ctx["is_htmx"]:
        ctx["side"] = side_queue()
        ctx["queue_count"] = ctx["side"]["count"]
    return templates.TemplateResponse(request, name, ctx, status_code=status_code)


def redirect(url: str, request: Request | None = None) -> RedirectResponse | HTMLResponse:
    """Redirect that also works for HTMX requests (which follow HX-Redirect)."""
    if request is not None and request.headers.get("hx-request") == "true":
        response = HTMLResponse("")
        response.headers["HX-Redirect"] = url
        return response
    return RedirectResponse(url, status_code=303)
