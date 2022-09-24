from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from ..diffusion.device import Device
from ..diffusion.device_pool import get_device
from .generator import generate_buffer, package_metadata, format_enum
from ..external_resource import get_image
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
    format: format_enum = format_enum.jpeg,
    guidance_scale: float = 7.5,
    strength: float = 0.75,
    num_inference_steps: int = 50,
    num_images: int = 1,
    device: Device = Depends(get_device),
):
    buffer, pipeline_config, args = generate_buffer(
        device,
        model_name="CompVis/stable-diffusion-v1-4",
        pipeline_name="imginpaint",
        guidance_scale=guidance_scale,
        strength=strength,
        num_inference_steps=num_inference_steps,
        num_images=num_images,
        prompt=prompt,
        init_image=get_image(image_uri),
        mask_image=get_image(mask_uri),
        format=format,
    )

    if format == format_enum.jpeg:
        return StreamingResponse(buffer, media_type="image/jpeg")

    if format == format_enum.png:
        return StreamingResponse(buffer, media_type="image/png")

    if format == format_enum.json:
        return package_metadata(buffer, pipeline_config, args)
