from mimetypes import init
import torch
import logging
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from torch.cuda.amp import autocast
from PIL import Image


class Pipelines():
    pipelines = {}

    def preload_pipelines(self, model_name, auth_token):
        logging.debug(f"Using device# {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")

        # this will preload the pieline into CPU memory
        # on demand they get swapped into gpu memory
        #
        # TODO #8 model the GPU as a class; including what pipeline is loaded and if it has a workload or not
        # TODO #9 implement img in-painting
        self.pipelines["txt2img"] = StableDiffusionPipeline.from_pretrained(
            model_name,
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=auth_token,
        )

        self.pipelines["img2img"] = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_name,
            revision="fp16", 
            torch_dtype=torch.float16,
            use_auth_token=auth_token,
        )

        self.pipelines["imginpaint"] = StableDiffusionInpaintPipeline.from_pretrained(
            model_name,
            revision="fp16", 
            torch_dtype=torch.float16,
            use_auth_token=auth_token,
        )


    # this does the actual image generation
    def get_txt2img(self, guidance_scale, num_inference_steps, num_images, height, width, prompt):
        logging.debug(f"Using device# {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
        if num_images > 9:
            raise Exception("The maximum number of images is 9")

        logging.info(f"Prompt is [{prompt}]")

        pipe = self.swap_pipelines("txt2img")

        # this can be done in a single pass to the pipeline but consumes a lot of memory and isn't much faster
        for i in range(num_images):
            with autocast():
                images = pipe(prompt, 
                    guidance_scale=guidance_scale, 
                    num_inference_steps=num_inference_steps,
                    height=height, 
                    width=width
                ).images

        return (post_process(num_images, images), pipe.config)


    def get_img2img(self, strength, guidance_scale, num_inference_steps, num_images, prompt, init_image):
        logging.debug(f"Using device# {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
        if num_images > 9:
            raise Exception("The maximum number of images is 9")

        logging.info(f"Prompt is [{prompt}]")

        pipe = self.swap_pipelines("img2img")

        # this can be done in a single pass to the pipeline but consumes a lot of memory and isn't much faster
        for i in range(num_images):
            with autocast():
                images = pipe(prompt, 
                    guidance_scale=guidance_scale, 
                    num_inference_steps=num_inference_steps,
                    init_image=init_image,
                    strength=strength
                ).images

        return (post_process(num_images, images), pipe.config)


    def get_imginpaint(self, strength, guidance_scale, num_inference_steps, num_images, prompt, init_image, mask_image):
        logging.debug(f"Using device# {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
        if num_images > 9:
            raise Exception("The maximum number of images is 9")

        logging.info(f"Prompt is [{prompt}]")

        pipe = self.swap_pipelines("imginpaint")

        # this can be done in a single pass to the pipeline but consumes a lot of memory and isn't much faster
        for i in range(num_images):
            with autocast():
                images = pipe(prompt, 
                    guidance_scale=guidance_scale, 
                    num_inference_steps=num_inference_steps,
                    init_image=init_image,
                    strength=strength,
                    mask_image=mask_image
                ).images

        return (post_process(num_images, images), pipe.config)


    def swap_pipelines(self, pipeline_name):
        with torch.no_grad():
            torch.cuda.empty_cache()

        self.pipelines[pipeline_name].to("cuda")

        return self.pipelines[pipeline_name]


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
