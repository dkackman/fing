from typing import Optional
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from ..diffusion.device import Device
from ..diffusion.device_pool import add_device_to_pool, remove_device_from_pool
from .generator import generate_buffer, package_metadata, image_format_enum, get_image
from .x_api_key import x_api_key_auth

img2img_router = APIRouter()


@img2img_router.get(
    "/img2img",
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
    format: image_format_enum = image_format_enum.jpeg,
    guidance_scale: float = 7.5,
    strength: float = 0.75,
    num_inference_steps: int = 25,
    num_images: int = 1,
    seed: Optional[int] = None,
    negative_prompt: Optional[str] = None,
    device: Device = Depends(remove_device_from_pool),
):
    try:
        buffer, pipeline_config, args = generate_buffer(
            device,
            model_name="runwayml/stable-diffusion-v1-5",
            pipeline_name="img2img",
            guidance_scale=guidance_scale,
            strength=strength,
            num_inference_steps=num_inference_steps,
            num_images=num_images,
            prompt=prompt,
            init_image=get_image(image_uri),
            format=format,
            seed=seed,
            negative_prompt=negative_prompt,
            revision="fp16",
        )
    finally:
        add_device_to_pool(device)

    if format == image_format_enum.jpeg:
        return StreamingResponse(buffer, media_type="image/jpeg")

    if format == image_format_enum.png:
        return StreamingResponse(buffer, media_type="image/png")

    if format == image_format_enum.json:
        args["image_uri"] = image_uri
        return package_metadata(buffer, pipeline_config, args)
