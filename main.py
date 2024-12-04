from fastapi import FastAPI
from starlette.requests import Request
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from routers import api

app = FastAPI(
    title="ML Course Project Api",
    description="Magankov K.S (IVT-301) © 2024",
    contact={
        "name": "Kirill Magankov",
        "url": "https://t.me/zntnaxbi_mk",
    },
    version="1.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.include_router(api.router, prefix="/api/v1")


def common_context():
    return {
        'has_header': True,
        'has_footer': True,
    }


@app.get("/", include_in_schema=False)
async def index(request: Request):
    context = {
        'title': 'Home',
        **common_context(),
    }

    return templates.TemplateResponse(
        request=request, name="index.html",
        context=context
    )
