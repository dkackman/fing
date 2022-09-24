import logging
import torch
from .service.web_app import create_app
from . import __version__
import uvicorn
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)
from .diffusion.device import Device
from .diffusion.pipelines import Pipelines
from .diffusion.device_pool import add_device

if not torch.cuda.is_available():
    raise Exception("CUDA not present. Quitting.")

app, settings = create_app()

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
