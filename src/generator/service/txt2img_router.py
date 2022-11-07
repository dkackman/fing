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

txt2img_router = APIRouter()


@txt2img_router.get(
    "/txt2img",
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
    format: image_format_enum = image_format_enum.jpeg,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    num_images: int = 1,
    height: int = 512,
    width: int = 512,
    use_ldm: bool = False,
    use_lpw: bool = False,
    use_composable: bool = False,
    seed: Optional[int] = None,
    negative_prompt: Optional[str] = None,
    weights: Optional[str] = None,
    device: Device = Depends(remove_device_from_pool),
):
    try:
        model_name = "runwayml/stable-diffusion-v1-5"
        custom_pipeline = None
        revision = "fp16"
        torch_dtype = torch.float16

        if use_composable:
            custom_pipeline = "composable_stable_diffusion"
            revision = "main"
        elif use_lpw:
            model_name = "hakurei/waifu-diffusion"
            custom_pipeline = "lpw_stable_diffusion"
        elif use_ldm:
            model_name = "CompVis/ldm-text2im-large-256"
            revision = "main"
            torch_dtype = torch.float32

        buffer, pipeline_config, args = generate_buffer(
            device,
            model_name=model_name,
            pipeline_name="txt2img",
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images=num_images,
            height=height,
            width=width,
            prompt=prompt,
            format=format,
            seed=seed,
            negative_prompt=negative_prompt,
            revision=revision,
            custom_pipeline=custom_pipeline,
            weights=weights,
            torch_dtype=torch_dtype,
        )
    finally:
        add_device_to_pool(device)

    if format == image_format_enum.jpeg:
        return StreamingResponse(buffer, media_type="image/jpeg")

    if format == image_format_enum.png:
        return StreamingResponse(buffer, media_type="image/png")

    if format == image_format_enum.json:
        return package_metadata(buffer, pipeline_config, args)
