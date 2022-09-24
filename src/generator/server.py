import logging
import torch
import uvicorn
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)
from . import __version__
from .settings import load_settings, resolve_path
from .log_setup import setup_logging
from .service.web_app import create_app
from .diffusion.device import Device
from .diffusion.pipelines import Pipelines
from .diffusion.device_pool import add_device

if not torch.cuda.is_available():
    raise Exception("CUDA not present. Quitting.")

settings = load_settings()
setup_logging(resolve_path(settings.log_filename), settings.log_level)

app = create_app(
    __version__, settings.x_api_key_enabled, settings.x_api_key_list
)

logging.debug(f"Torch version {torch.__version__}")


@app.on_event("startup")
async def startup_event():
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


if __name__ == "__main__":
    uvicorn.run("generator.server:app", host=settings.host, port=settings.port)
