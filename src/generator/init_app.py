from .settings import (
    Settings,
    settings_exist,
    save_settings,
    get_settings_full_path,
    load_settings,
    load_settings,
    resolve_path,
    settings_exist,    
)

import asyncio
import logging
from .log_setup import setup_logging
from diffusers import DiffusionPipeline
import torch


async def init():
    logging.info("init_app")

    overwrite = False
    if settings_exist():
        print("A settings file already exists.")
        overwrite = input("Do you want to overwrite these settings? (y/N): ").strip().lower()
        if len(overwrite) > 0 or overwrite.startswith("y"):
            overwrite = True

    if not settings_exist() or overwrite:
        # this path is legacy since dafe defaults get created at startup - might remove later
        settings = Settings()

        print("Provide the following details for the intial configuration:\n")
        token = input("Huggingface API token: ").strip()
        if len(token) == 0:
            print("A Huggingface API token is required.")
            return

        host = input("Service host (localhost): ").strip()
        host = "localhost" if len(host) == 0 else host

        port = input("Service port (9147): ").strip()
        port = 9147 if len(port) == 0 else int(port)

        sdaas_token = input("sdaas token: ").strip()

        sdaas_uri = input("sdaas uri (http://fing.kackman.net:9511): ").strip()
        sdaas_uri = "http://fing.kackman.net:9511" if len(sdaas_uri) == 0 else sdaas_uri

        settings.huggingface_token = token
        settings.host = host
        settings.port = port
        settings.sdaas_token = sdaas_token
        settings.sdaas_uri = sdaas_uri

        save_settings(settings)
        print(f"Configuration saved to {get_settings_full_path()}")
    
    settings = load_settings()
    setup_logging(resolve_path(settings.log_filename), settings.log_level)
    logging.debug(f"Torch version {torch.__version__}")
    print("Preloading pipelines. This may take awhile...")

    models = [
        ("stabilityai/stable-diffusion-2", "fp16", None),
        ("stabilityai/stable-diffusion-2-base", "fp16", None),
        ("stabilityai/stable-diffusion-2-inpainting", "fp16", None),
        ("stabilityai/stable-diffusion-x4-upscaler", "fp16", None),
        ("nitrosocke/Future-Diffusion", "main", None),
        ("prompthero/openjourney", "main", None),
        ("runwayml/stable-diffusion-v1-5", "fp16", None),
        ("runwayml/stable-diffusion-v1-5", "main", "composable_stable_diffusion"),
        ("runwayml/stable-diffusion-inpainting", "fp16", None),
        ("CompVis/ldm-celebahq-256", "main", None),
        ("CompVis/ldm-text2im-large-256", "main", None),
        ("hakurei/waifu-diffusion", "fp16", "lpw_stable_diffusion"),
    ]

    # this makes sure that all of the diffusers are downloaded and cached
    for model in models:
        DiffusionPipeline.from_pretrained(
            model[0],
            use_auth_token=settings.huggingface_token,
            device_map="auto",
            revision=model[1],
            torch_dtype=torch.float16,
            custom_pipeline=model[2],
        )
    print("done")


asyncio.run(init())
