import torch
import logging
from diffusers import StableDiffusionPipeline
from torch.cuda.amp import autocast
from PIL import Image


def load_model(model, auth_token):
    logging.debug(f"Using device# {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
    global pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        model,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=auth_token,
    )

    pipe.to("cuda")  # Run on GPU


def generate(guidance_scale, num_inference_steps, num_images, height, width, prompt):
    logging.debug(f"Using device# {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
    return get_image(pipe, guidance_scale, num_inference_steps, num_images, height, width, prompt)


# this does the actual image generation
def get_image(pipe, guidance_scale, num_inference_steps, num_images, height, width, prompt):
    if num_images > 9:
        raise Exception("The maximum number of images is 9")

    logging.info(f"Prompt is [{prompt}]")

    images = []
    # this can be done in a single pass to the pipeline but consumes a lot of memory and isn't much faster
    for i in range(num_images):
        with autocast():
            results = pipe(prompt, 
                guidance_scale=guidance_scale, 
                num_inference_steps=num_inference_steps,
                height=height, 
                width=width
            )["images"]
            for image in results:
                images.append(image)

    if num_images == 1:
        image = images[0]
    elif num_images == 2:
        image = image_grid(images, 1, 2)
    elif num_images <= 4:
        image = image_grid(images, 2, 2)
    elif num_images <= 6:
        image = image_grid(images, 3, 2)       
    elif num_images <= 9:
        image = image_grid(images, 3, 3)   

    return image


def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))

    return grid

