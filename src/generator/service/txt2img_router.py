from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from ..diffusion.device import Device
from ..diffusion.device_pool import get_device
from .generator import generate_buffer, package_metadata
from .x_api_key import x_api_key_auth

txt2img_router = APIRouter()


@txt2img_router.get("/txt2img", dependencies=[Depends(x_api_key_auth)])
def get_txt2img(
    prompt: str,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    num_images: int = 1,
    height=512,
    width=512,
    device: Device = Depends(get_device),
):
    buffer, pipeline_config, args = generate_buffer(
        device,
        pipeline_name="txt2img",
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images=num_images,
        height=height,
        width=width,
        prompt=prompt,
    )

    return StreamingResponse(buffer, media_type="image/jpeg")


@txt2img_router.get("/txt2img_metadata", dependencies=[Depends(x_api_key_auth)])
def get_txt2img_metadata(
    prompt: str,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    num_images: int = 1,
    height=512,
    width=512,
    device: Device = Depends(get_device),
):
    buffer, pipeline_config, args = generate_buffer(
        device,
        pipeline_name="txt2img",
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images=num_images,
        height=height,
        width=width,
        prompt=prompt,
    )

    return package_metadata(buffer, pipeline_config, args)
