import logging
import torch
import uvicorn
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
from .diffusion.device_pool import add_device_to_pool
from diffusers import DiffusionPipeline

if not settings_exist():
    print("Initializing settings with defaults")
    save_settings(Settings())

settings = load_settings()


app = create_app(__version__, settings.x_api_key_enabled, settings.x_api_key_list)


async def do_setup():
    setup_logging(resolve_path(settings.log_filename), settings.log_level)
    logging.debug(f"Torch version {torch.__version__}")

    models = [
        ("CompVis/stable-diffusion-v1-4", "fp16"),
        ("CompVis/ldm-celebahq-256", "main"),
        ("runwayml/stable-diffusion-inpainting", "fp16"),
        ("CompVis/ldm-text2im-large-256", "main"),
    ]

    # this makes sure that all of the diffusers are downloaded and cached
    for model in models:
        DiffusionPipeline.from_pretrained(
            model[0],
            use_auth_token=settings.huggingface_token,
            device_map="auto",
            revision=model[1],
            )


@app.on_event("startup")
async def startup_event():
    if not torch.cuda.is_available():
        raise Exception("CUDA not present. Quitting.")

    await do_setup()

    for i in range(0, torch.cuda.device_count()):
        logging.info(f"Adding cuda device {i} - {torch.cuda.get_device_name(i)}")
        add_device_to_pool(Device(i, settings.huggingface_token))


if __name__ == "__main__":
    uvicorn.run("generator.server:app", host=settings.host, port=settings.port)
