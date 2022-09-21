import logging
import torch
from fastapi import FastAPI
from ..diffusion.device import Device
from ..diffusion.pipelines import Pipelines
from .info_router import info_router
from .txt2img_router import txt2img_router
from .img2img_router import img2img_router
from .imginpaint_router import imginpaint_router
from ..diffusion.device_pool import add_device
from .. import __version__


def create_app(model_name, auth_token, model_cache_dir):
    logging.debug(f"Torch version {torch.__version__}")

    pipelines = Pipelines(model_name, model_cache_dir).preload_pipelines(
        auth_token, ["imginpaint"]
    )
    add_device(Device(pipelines))

    app = FastAPI(title="stable-diffusion service", version=__version__)

    app.include_router(info_router)
    app.include_router(txt2img_router)
    app.include_router(img2img_router)
    app.include_router(imginpaint_router)

    return app
