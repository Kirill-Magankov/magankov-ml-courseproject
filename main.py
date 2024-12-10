import base64
import os

import dotenv
from fastapi import FastAPI, File
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from typing_extensions import Annotated

from helpers import get_metrics
from routers import api
from routers.api import api_classification

dotenv.load_dotenv()

ALLOWED_HOST = os.getenv("ALLOWED_HOST", '*')

app = FastAPI(
    title="ML Course Project Api",
    description="Magankov K.S (IVT-301) © 2024",
    contact={
        "name": "Kirill Magankov",
        "url": "https://t.me/zntnaxbi_mk",
    },
    version="1.0"
)

app.add_middleware(
    TrustedHostMiddleware,  # noqa
    allowed_hosts=ALLOWED_HOST.split(',')
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.include_router(api.router, prefix="/api/v1")

metrics = get_metrics()


def common_context():
    return {
        'has_header': True,
        'has_footer': True,
    }


@app.get("/", include_in_schema=False)
async def index(request: Request):
    context = {
        'title': 'Home',
        'metrics': metrics,
        **common_context(),
    }

    return templates.TemplateResponse(
        request=request, name="index.html",
        context=context
    )


@app.post('/', include_in_schema=False)
async def index_post(request: Request, image: Annotated[bytes, File()]):
    if not image: return RedirectResponse(url='/', status_code=301)

    context = {
        'title': 'Home | Results',
        'prediction': await api_classification(image),
        'metrics': metrics,
        'image': base64.b64encode(image).decode(),
        **common_context(),
    }

    return templates.TemplateResponse(
        request=request, name="index.html",
        context=context
    )
