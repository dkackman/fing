from typing import Optional
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from ..diffusion.device import Device
from ..diffusion.device_pool import add_device_to_pool, remove_device_from_pool
from .generator import (
    generate_buffer,
    package_metadata,
    image_format_enum,
    PackageMetaDataModel,
)
from .x_api_key import x_api_key_auth
import torch


random_face_router = APIRouter()


@random_face_router.get(
    "/random_face",
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
        },
    },
    response_model=PackageMetaDataModel,
)
def get_img(
    format: image_format_enum = image_format_enum.jpeg,
    num_inference_steps: int = 50,
    num_images: int = 1,
    seed: Optional[int] = None,
    device: Device = Depends(remove_device_from_pool),
):
    try:
        buffer, pipeline_config, args = generate_buffer(
            device,
            model_name="CompVis/ldm-celebahq-256",
            pipeline_name="faces",
            num_inference_steps=num_inference_steps,
            num_images=num_images,
            format=format,
            seed=seed,
            revision="main",
            torch_dtype=torch.float32
        )
    finally:
        add_device_to_pool(device)

    if format == image_format_enum.jpeg:
        return StreamingResponse(buffer, media_type="image/jpeg")

    if format == image_format_enum.png:
        return StreamingResponse(buffer, media_type="image/png")

    if format == image_format_enum.json:
        return package_metadata(buffer, pipeline_config, args)
