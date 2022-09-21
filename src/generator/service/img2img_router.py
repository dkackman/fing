from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from ..diffusion.device import Device
from ..diffusion.device_pool import get_device
from .generator import generate_buffer, package_metadata
from ..external_resource import get_image

img2img_router = APIRouter()


@img2img_router.get("/img2img")
def get_img2img(
    prompt: str,
    image_uri: str,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    num_images: int = 1,
    device: Device = Depends(get_device),
):
    buffer, pipeline_config, args = generate_buffer(
        device,
        pipeline_name="img2img",
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images=num_images,
        prompt=prompt,
        init_image=get_image(image_uri),
    )

    return StreamingResponse(buffer, media_type="image/jpeg")


@img2img_router.get("/img2img_metadata")
def get_img2img_metadata(
    prompt: str,
    image_uri: str,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    num_images: int = 1,
    device: Device = Depends(get_device),
):
    buffer, pipeline_config, args = generate_buffer(
        device,
        pipeline_name="img2img",
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images=num_images,
        prompt=prompt,
        init_image=get_image(image_uri),
    )

    # don't serialize the image, just the uri as part of the parameters json
    # the buffer is serialized as base64
    args.pop("init_image")
    args["image_uri"] = image_uri
    return package_metadata(buffer, pipeline_config, args)
