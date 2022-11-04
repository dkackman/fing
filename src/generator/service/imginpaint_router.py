from typing import Optional
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from ..diffusion.device import Device
from ..diffusion.device_pool import add_device_to_pool, remove_device_from_pool
from .generator import generate_buffer, package_metadata, image_format_enum, get_image
from .x_api_key import x_api_key_auth

imginpaint_router = APIRouter()


@imginpaint_router.get(
    "/imginpaint",
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
    prompt: str,
    image_uri: str,
    mask_uri: str,
    format: image_format_enum = image_format_enum.jpeg,
    guidance_scale: float = 7.5,
    strength: float = 0.75,
    num_inference_steps: int = 50,
    num_images: int = 1,
    seed: Optional[int] = None,
    negative_prompt: Optional[str] = None,
    device: Device = Depends(remove_device_from_pool),
):
    try:
        buffer, pipeline_config, args = generate_buffer(
            device,
            model_name="runwayml/stable-diffusion-inpainting",
            pipeline_name="imginpaint",
            guidance_scale=guidance_scale,
            strength=strength,
            num_inference_steps=num_inference_steps,
            num_images=num_images,
            prompt=prompt,
            image=get_image(image_uri),
            mask_image=get_image(mask_uri),
            format=format,
            seed=seed,
            negative_prompt=negative_prompt,
        )
    finally:
        add_device_to_pool(device)

    if format == image_format_enum.jpeg:
        return StreamingResponse(buffer, media_type="image/jpeg")

    if format == image_format_enum.png:
        return StreamingResponse(buffer, media_type="image/png")

    if format == image_format_enum.json:
        # don't serialize these two in the metadata
        args["image_uri"] = image_uri
        args["mask_uri"] = image_uri
        return package_metadata(buffer, pipeline_config, args)
