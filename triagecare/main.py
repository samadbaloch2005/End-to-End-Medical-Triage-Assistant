"""Application entry point.

    uvicorn triagecare.main:app --reload
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from urllib.parse import quote

from triagecare import __version__
from triagecare.config import get_settings
from triagecare.deps import LoginRequired
from triagecare.routers import admin, api, appointments, auth, dashboard, patients, queue, visits
from triagecare.services.triage import get_model
from triagecare.web import render

log = logging.getLogger("triagecare")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.is_production and settings.secret_key == "dev-only-change-me":
        raise RuntimeError("Set SECRET_KEY before running in production.")
    if not get_model().available:
        log.warning("No trained model at %s. Run `python -m triagecare.cli setup`.", settings.model_path)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=f"{settings.app_name} API",
        version=__version__,
        description="Emergency department triage: patients, appointments, queue and ML risk scoring.",
        lifespan=lifespan,
    )
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.secret_key,
        session_cookie="triagecare_session",
        max_age=60 * 60 * 12,  # one shift
        same_site="lax",
        https_only=settings.is_production,
    )
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

    for module in (auth, dashboard, patients, appointments, visits, queue, admin, api):
        app.include_router(module.router)

    @app.exception_handler(LoginRequired)
    async def _login_required(request: Request, exc: LoginRequired):
        target = request.url.path
        if request.url.query:
            target += "?" + request.url.query
        url = "/login" if target == "/" else f"/login?next={quote(target)}"
        if request.headers.get("hx-request") == "true":
            return JSONResponse({}, headers={"HX-Redirect": url})
        return RedirectResponse(url, status_code=303)

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException):
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        return render(
            request,
            "error.html",
            {"code": exc.status_code, "detail": exc.detail, "user": None},
            status_code=exc.status_code,
        )

    @app.exception_handler(404)
    async def _not_found(request: Request, exc):
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        return render(request, "error.html", {"code": 404, "detail": "Page not found", "user": None}, 404)

    return app


app = create_app()
