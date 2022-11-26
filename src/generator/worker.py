from .diffusion.device import Device
from .diffusion.device_pool import add_device_to_pool, remove_device_from_pool
from .service.generator import (
    generate_buffer,
    image_format_enum,
)
import torch
import asyncio
import logging
from .settings import (
    Settings,
    load_settings,
    resolve_path,
    settings_exist,
    save_settings,
)
from .log_setup import setup_logging
import base64
import json
import requests

if not settings_exist():
    print("Initializing settings with defaults")
    save_settings(Settings())

settings = load_settings()


async def run_worker():
    await do_setup()
    await startup_event()

    logging.info("worker")

    while True:
        try:
            uri = f"{settings.sdaas_uri}/api"
            response = requests.get(
                f"{uri}/work",
                headers={
                    "Content-type": "application/json",
                    "Authorization": f"Bearer {settings.sdaas_token}",
                },
            )
            response_dict = response.json()

            for job in response_dict["jobs"]:
                revision = "fp16"
                if (
                    job["model_name"] == "nitrosocke/Future-Diffusion"
                    or job["model_name"] == "prompthero/openjourney"
                ):
                    revision = "main"

                torch_dtype = torch.float16
                device = remove_device_from_pool()
                try:
                    buffer, pipeline_config, args = generate_buffer(
                        device,
                        prompt=job["prompt"],
                        negative_prompt=job["negative_prompt"],
                        model_name=job["model_name"],
                        pipeline_name="txt2img",
                        format=image_format_enum.jpeg,
                        revision=revision,
                        torch_dtype=torch_dtype,
                        num_inference_steps=25,
                    )

                    result = {
                        "id": job["id"],
                        "model_name": job["model_name"],
                        "prompt": job["prompt"],
                        "negative_prompt": job["negative_prompt"],
                        "contentType": "image/jpeg",
                        "blob": base64.b64encode(buffer.getvalue()).decode("UTF-8"),
                    }
                    requests.post(
                        f"{uri}/results",
                        data=json.dumps(result),
                        headers={
                            "Content-type": "application/json",
                            "Authorization": f"Bearer {settings.sdaas_token}",
                        },
                    )

                except Exception as e:
                    print(e)
                finally:
                    add_device_to_pool(device)
        except Exception as e:
            print(e)  # this is if the work queue endpoint is unavailable
            await asyncio.sleep(60)

        await asyncio.sleep(10)


async def do_setup():
    setup_logging(resolve_path(settings.log_filename), settings.log_level)
    logging.debug(f"Torch version {torch.__version__}")


async def startup_event():
    if not torch.cuda.is_available():
        raise Exception("CUDA not present. Quitting.")

    await do_setup()

    for i in range(0, torch.cuda.device_count()):
        logging.info(f"Adding cuda device {i} - {torch.cuda.get_device_name(i)}")
        add_device_to_pool(Device(i, settings.huggingface_token))


if __name__ == "__main__":
    asyncio.run(run_worker())
