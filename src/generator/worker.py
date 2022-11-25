from .diffusion.device import Device
from .diffusion.device_pool import add_device_to_pool, remove_device_from_pool
from .service.generator import (
    generate_buffer,
    image_format_enum,
)
import torch
import asyncio
import logging
from azure.storage.queue import QueueClient
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from .settings import (
    Settings,
    load_settings,
    resolve_path,
    settings_exist,
    save_settings,
)
from .log_setup import setup_logging
import json

connect_str = "DefaultEndpointsProtocol=https;AccountName=sdaasdev;AccountKey=mxyfLArLl60gAJyxhWVE9D4oyW/MULozcA88BirRW/NDy36rJdFVb/YUqpWcrEvrvyW6DEsEQHV9+AStIPC/Jw==;EndpointSuffix=core.windows.net"

if not settings_exist():
    print("Initializing settings with defaults")
    save_settings(Settings())

settings = load_settings()


async def run_worker():
    await do_setup()
    await startup_event()

    logging.info("worker")
    queue = QueueClient.from_connection_string(
        conn_str=connect_str,
        queue_name="work",
    )

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    while True:
        messages = queue.receive_messages()

        for message in messages:
            job = json.loads(message.content)  # type: ignore
            model_name = "runwayml/stable-diffusion-v1-5"
            revision = "fp16"
            torch_dtype = torch.float16
            device = remove_device_from_pool()
            try:
                buffer, pipeline_config, args = generate_buffer(
                    device,
                    prompt=job["prompt"],
                    model_name=model_name,
                    pipeline_name="txt2img",
                    format=image_format_enum.jpeg,
                    revision=revision,
                    torch_dtype=torch_dtype,
                    num_inference_steps=25,
                )
                blob_client = blob_service_client.get_blob_client(container="results", blob=job["id"])
                blob_client.upload_blob(buffer)
                queue.delete_message(message.id, message.pop_receipt)

            except Exception as e:
                print(e)
            finally:
                add_device_to_pool(device)

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
