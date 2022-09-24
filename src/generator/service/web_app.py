from fastapi import FastAPI
from .info_router import info_router
from .txt2img_router import txt2img_router
from .img2img_router import img2img_router
from .imginpaint_router import imginpaint_router
from .random_face_router import random_face_router
from .compose_router import compose_router
from .x_api_key import enable_x_api_keys


def create_app(version, x_api_key_enabled, x_api_key_list):
    if x_api_key_enabled:
        enable_x_api_keys(x_api_key_list)

    app = FastAPI(
        title="stable-diffusion service",
        version=version,
        description="Rest interface to stable-diffusion image generation",
        license_info={
            "name": "Apache 2.0",
            "url": "http://www.apache.org/licenses/LICENSE-2.0.html",
        },
        contact={
            "name": "dkackman",
            "url": "https://github.com/dkackman/fing",
        },
    )

    app.include_router(info_router)
    app.include_router(txt2img_router)
    app.include_router(img2img_router)
    app.include_router(imginpaint_router)
    app.include_router(random_face_router)
    app.include_router(compose_router)

    return app
