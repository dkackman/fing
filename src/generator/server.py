import logging
import torch
import uvicorn
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)
from . import __version__
from .settings import (
    Settings,
    load_settings,
    resolve_path,
    settings_exist,
    save_settings,
)
from .log_setup import setup_logging
from .service.web_app import create_app
from .diffusion.device import Device
from .diffusion.pipeline_cache import PipelineCache
from .diffusion.device_pool import add_device_to_pool


if not settings_exist():
    print("Initializing settings with defaults")
    save_settings(Settings())

settings = load_settings()


app = create_app(__version__, settings.x_api_key_enabled, settings.x_api_key_list)


async def do_setup():
    setup_logging(resolve_path(settings.log_filename), settings.log_level)
    logging.debug(f"Torch version {torch.__version__}")

    pipeline_cache = PipelineCache(settings.model_cache_dir)

    pipeline_cache.preload(
        settings.huggingface_token,
        "CompVis/ldm-celebahq-256",
        {
            "faces": DiffusionPipeline,
        },
        "main",
        enable_attention_slicing=False,
    )
    pipeline_cache.preload(
        settings.huggingface_token,
        "CompVis/stable-diffusion-v1-4",
        {
            "txt2img": StableDiffusionPipeline,
            "img2img": StableDiffusionImg2ImgPipeline,
        },
        "fp16",
    )
    pipeline_cache.preload(
        settings.huggingface_token,
        "runwayml/stable-diffusion-inpainting",
        {
            "imginpaint": StableDiffusionInpaintPipeline,
        },
        "fp16",
    )
    pipeline_cache.preload(
        settings.huggingface_token,
        "CompVis/ldm-text2im-large-256",
        {
            "txt2img": DiffusionPipeline,
        },
        "main",
        enable_attention_slicing=False,
    )

    return pipeline_cache


@app.on_event("startup")
async def startup_event():
    if not torch.cuda.is_available():
        raise Exception("CUDA not present. Quitting.")

    pipeline_cache = await do_setup()

    for i in range(0, torch.cuda.device_count()):
        logging.info(f"Adding cuda device {i} - {torch.cuda.get_device_name(i)}")
        add_device_to_pool(Device(i, pipeline_cache))


if __name__ == "__main__":
    uvicorn.run("generator.server:app", host=settings.host, port=settings.port)
