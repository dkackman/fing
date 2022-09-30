from typing import Optional
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from ..diffusion.device import Device
from ..diffusion.device_pool import remove_device_from_pool
from .generator import (
    generate_buffer,
    package_metadata,
    format_enum,
    PackageMetaDataModel,
)
from .x_api_key import x_api_key_auth

compose_router = APIRouter()


@compose_router.get(
    "/compose",
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
    prompt: str,
    format: format_enum = format_enum.jpeg,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    num_images: int = 1,
    height: int = 512,
    width: int = 512,
    seed: Optional[int] = None,
    device: Device = Depends(remove_device_from_pool),
):
    buffer, pipeline_config, args = generate_buffer(
        device,
        model_name="CompVis/stable-diffusion-v1-4",
        pipeline_name="compose",
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images=num_images,
        height=height,
        width=width,
        prompt=prompt,
        format=format,
        seed=seed,
    )

    if format == format_enum.jpeg:
        return StreamingResponse(buffer, media_type="image/jpeg")

    if format == format_enum.png:
        return StreamingResponse(buffer, media_type="image/png")

    if format == format_enum.json:
        return package_metadata(buffer, pipeline_config, args)
