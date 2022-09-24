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
from .x_api_key import enable_x_api_keys
from .settings import load_settings, resolve_path
from concurrent_log_handler import ConcurrentRotatingFileHandler
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)


def create_app():
    settings = load_settings()
    setup_logging(settings.log_filename, settings.log_level)

    logging.debug(f"Torch version {torch.__version__}")

    pipelines = Pipelines(settings.model_cache_dir).preload_pipelines(
        settings.huggingface_token,
        "CompVis/stable-diffusion-v1-4",
        {
            "txt2img": StableDiffusionPipeline,
            "img2img": StableDiffusionImg2ImgPipeline,
            "imginpaint": StableDiffusionInpaintPipeline,
        },
    )
    add_device(Device(0, pipelines))

    if settings.x_api_key_enabled:
        enable_x_api_keys(settings.x_api_key_list)

    app = FastAPI(
        title="stable-diffusion service",
        version=__version__,
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

    return app, settings


def setup_logging(log_path, log_level):
    log_path = resolve_path(log_path)
    log_date_format = "%Y-%m-%dT%H:%M:%S"

    logger = logging.getLogger()

    handler = ConcurrentRotatingFileHandler(
        log_path, "a", maxBytes=50 * 1024 * 1024, backupCount=7
    )
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt=log_date_format
        )
    )

    if log_level == "CRITICAL":
        logger.setLevel(logging.CRITICAL)
    elif log_level == "ERROR":
        logger.setLevel(logging.ERROR)
    elif log_level == "WARNING":
        logger.setLevel(logging.WARNING)
    elif log_level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.addHandler(handler)
