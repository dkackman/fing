import torch
import logging
from torch.cuda.amp import autocast
from PIL import Image


# TODO #8 model the GPU as a class; including what pipeline is loaded and if it has a workload or not
class Device():
    pipelines = None

    def __init__(self, pipelines) -> None:
        self.pipelines = pipelines


    # this does the actual image generation
    def get_txt2img(self, guidance_scale, num_inference_steps, num_images, height, width, prompt):
        log_device()
        if num_images > 9:
            raise Exception("The maximum number of images is 9")

        logging.info(f"Prompt is [{prompt}]")

        pipe = self.pipelines.get_pipeline("txt2img")

        # this can be done in a single pass to the pipeline but consumes a lot of memory and isn't much faster
        for i in range(num_images):
            with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                images = pipe(prompt, 
                    guidance_scale=guidance_scale, 
                    num_inference_steps=num_inference_steps,
                    height=height, 
                    width=width
                ).images

        return (post_process(num_images, images), pipe.config)


    def get_img2img(self, strength, guidance_scale, num_inference_steps, num_images, prompt, init_image):
        log_device()
        if num_images > 9:
            raise Exception("The maximum number of images is 9")

        logging.info(f"Prompt is [{prompt}]")

        pipe = self.pipelines.get_pipeline("img2img")

        # this can be done in a single pass to the pipeline but consumes a lot of memory and isn't much faster
        for i in range(num_images):
            with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                images = pipe(prompt, 
                    guidance_scale=guidance_scale, 
                    num_inference_steps=num_inference_steps,
                    init_image=init_image,
                    strength=strength
                ).images

        return (post_process(num_images, images), pipe.config)


    def get_imginpaint(self, strength, guidance_scale, num_inference_steps, num_images, prompt, init_image, mask_image):
        log_device()
        if num_images > 9:
            raise Exception("The maximum number of images is 9")

        logging.info(f"Prompt is [{prompt}]")

        pipe = self.pipelines.get_pipeline("imginpaint")

        # this can be done in a single pass to the pipeline but consumes a lot of memory and isn't much faster
        for i in range(num_images):
            with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                images = pipe(prompt, 
                    guidance_scale=guidance_scale, 
                    num_inference_steps=num_inference_steps,
                    init_image=init_image,
                    strength=strength,
                    mask_image=mask_image
                ).images

        return (post_process(num_images, images), pipe.config)


def log_device():
    if torch.cuda.is_available():
        logging.debug(f"Using device# {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        logging.debug("Using cpu")


def post_process(num_images, images_list):
    if num_images == 1:
        image = images_list[0]
    elif num_images == 2:
        image = image_grid(images_list, 1, 2)
    elif num_images <= 4:
        image = image_grid(images_list, 2, 2)
    elif num_images <= 6:
        image = image_grid(images_list, 2, 3)       
    elif num_images <= 9:
        image = image_grid(images_list, 3, 3)   

    return image


def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))

    return grid
