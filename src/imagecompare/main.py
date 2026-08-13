"""FastAPI application entry point.

Run with:
    uvicorn imagecompare.main:app --reload
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from imagecompare.api.routes import router as api_router
from imagecompare.config import settings

BASE_DIR = Path(__file__).parent
WEB_DIR = BASE_DIR / "web"

app = FastAPI(title="Image Compare", version="0.1.0")

app.include_router(api_router)
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

templates = Jinja2Templates(directory=WEB_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "default_visual_threshold": settings.default_visual_threshold,
            "default_filename_threshold": settings.default_filename_threshold,
            "default_extensions": " ".join(settings.default_extensions),
        },
    )


def run() -> None:
    """Console-script entry point (see pyproject.toml)."""
    import uvicorn

    uvicorn.run("imagecompare.main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    run()
