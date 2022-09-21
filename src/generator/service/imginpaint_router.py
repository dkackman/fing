from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from ..diffusion.device import Device
from ..diffusion.device_pool import get_device
from .generator import generate_buffer, package_metadata
from ..external_resource import get_image

imginpaint_router = APIRouter()


@imginpaint_router.get("/imginpaint")
def get_imginpaint(
    prompt: str,
    image_uri: str,
    mask_uri: str,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    num_images: int = 1,
    device: Device = Depends(get_device),
):
    buffer, pipeline_config, args = generate_buffer(
        device,
        pipeline_name="imginpaint",
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images=num_images,
        prompt=prompt,
        init_image=get_image(image_uri),
        mask_image=get_image(mask_uri),
    )

    return StreamingResponse(buffer, media_type="image/jpeg")


@imginpaint_router.get("/imginpaint_metadata")
def get_imginpaint_metadata(
    prompt: str,
    image_uri: str,
    mask_uri: str,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    num_images: int = 1,
    device: Device = Depends(get_device),
):
    buffer, pipeline_config, args = generate_buffer(
        device,
        pipeline_name="imginpaint",
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images=num_images,
        prompt=prompt,
        init_image=get_image(image_uri),
        mask_image=get_image(mask_uri),
    )

    # don't serialize the image, just the uri as part of the parameters json
    # the buffer is serialized as base64
    args.pop("init_image")
    args.pop("mask_image")
    args["image_uri"] = image_uri
    args["mask_uri"] = mask_uri
    return package_metadata(buffer, pipeline_config, args)
