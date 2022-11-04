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
from PIL import Image
import io

experimental_router = APIRouter()


@experimental_router.get(
    "/experiment",
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
    height: int = 512,
    width: int = 512,
    strength: float = 0.55,
    seed: Optional[int] = None,
    device: Device = Depends(remove_device_from_pool),
):
    try:
        steps = prompt.split("|")
        steps.reverse()
        starting_prompt = steps.pop()
        init_image, pipeline_config, args = generate_buffer(
            device,
            model_name="CompVis/stable-diffusion-v1-4",
            pipeline_name="txt2img",
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            prompt=starting_prompt,
            format=format,
            seed=seed,
        )
    finally:
        add_device_to_pool(device)

    image_list = []

    next_buffer = init_image
    next_image = Image.open(next_buffer)
    image_list.append(next_image)
    for next_prompt in steps:
        next_buffer, _pipeline_config, _args = generate_buffer(
            device,
            model_name="CompVis/stable-diffusion-v1-4",
            pipeline_name="img2img",
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            prompt=next_prompt,
            format=format,
            init_image=next_image,
            strength=strength,
        )
        next_image = Image.open(next_buffer)
        image_list.append(next_image)

    buffer = image_to_buffer(post_process(image_list), "JPEG")

    if format == image_format_enum.jpeg:
        return StreamingResponse(buffer, media_type="image/jpeg")

    if format == image_format_enum.png:
        return StreamingResponse(buffer, media_type="image/png")

    if format == image_format_enum.json:
        return package_metadata(buffer, pipeline_config, args)


def image_to_buffer(image, format):
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)

    return buffer


def post_process(image_list):
    num_images = len(image_list)
    if num_images == 1:
        image = image_list[0]
    elif num_images == 2:
        image = image_grid(image_list, 1, 2)
    elif num_images <= 4:
        image = image_grid(image_list, 2, 2)
    elif num_images <= 6:
        image = image_grid(image_list, 2, 3)
    elif num_images <= 9:
        image = image_grid(image_list, 3, 3)

    return image


def image_grid(image_list, rows, cols):
    w, h = image_list[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(image_list):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid
