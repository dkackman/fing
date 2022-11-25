from typing import Optional
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from ..diffusion.device import Device
from ..diffusion.device_pool import add_device_to_pool, remove_device_from_pool
from .generator import generate_buffer, package_metadata, image_format_enum, get_image
from .x_api_key import x_api_key_auth
import torch

imgupscale_router = APIRouter()


@imgupscale_router.get(
    "/imgupscale",
    dependencies=[Depends(x_api_key_auth)],
    tags=["Stable Diffusion"],
    responses={
        200: {
            "content": {
                "image/jpeg": {},
                "image/png": {},
                "application/json": {},
            },
            "description": "The generated image.",
        }
    },
)
def get_img(
    image_uri: str,
    prompt: str = "",
    format: image_format_enum = image_format_enum.jpeg,
    num_inference_steps: int = 25,
    device: Device = Depends(remove_device_from_pool),
):
    try:
        buffer, pipeline_config, args = generate_buffer(
            device,
            model_name="stabilityai/stable-diffusion-x4-upscaler",
            prompt=prompt,
            pipeline_name="imgupscale",
            num_inference_steps=num_inference_steps,
            image=get_image(image_uri),
            format=format,
            revision="fp16",
        )
    finally:
        add_device_to_pool(device)

    if format == image_format_enum.jpeg:
        return StreamingResponse(buffer, media_type="image/jpeg")

    if format == image_format_enum.png:
        return StreamingResponse(buffer, media_type="image/png")

    if format == image_format_enum.json:
        # don't serialize this in the metadata
        args["image_uri"] = image_uri
        return package_metadata(buffer, pipeline_config, args)
