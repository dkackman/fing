from typing import Optional
import torch
import logging
from PIL import Image
from threading import Lock
from diffusers import DiffusionPipeline


class Device:
    device_id: int
    mutex: Lock

    def __init__(self, device_id: int, auth_token) -> None:
        self.device_id = device_id
        self.auth_token = auth_token
        self.mutex = Lock()

    def __call__(self, **kwargs):
        if not self.mutex.acquire(False):
            logging.error(f"Device {self.device_id} is busy but got invoked.")
            raise Exception("busy")

        try:
            num_images = kwargs.pop("num_images", 1)
            if num_images > 4:
                raise Exception("The maximum number of images is 4")

            if "prompt" in kwargs:
                logging.info(f"Prompt is {kwargs['prompt']}")
            self.log_device()

            pipeline = self.get_pipeline(
                kwargs.pop("model_name"),
                kwargs.pop("revision"),
                kwargs.pop("custom_pipeline", None),
                kwargs.pop("torch_dtype", torch.float16)
            )

            # this allows reproducability
            seed: Optional[int] = kwargs.pop("seed", None)
            if seed is None:
                seed = torch.seed()
            torch.manual_seed(seed)

            image_list = []
            nsfw_count = 0
            # this can be done in a single pass to the pipeline but consumes a lot of memory and isn't much faster
            for i in range(num_images):
                p = pipeline(**kwargs)
                if (
                    hasattr(p, "nsfw_content_detected")
                    and p.nsfw_content_detected[0] == True
                ):
                    logging.info(f"NSFW found in image {i}")
                    nsfw_count = nsfw_count + 1

                image_list.append(p.images[0])

            # if all the images are nsfw raise error as they will all be blank
            if len(image_list) == nsfw_count:
                raise Exception("NSFW")

            pipeline.config["seed"] = seed
            pipeline.config["class_name"] = pipeline.config["_class_name"]
            pipeline.config["diffusers_version"] = pipeline.config["_diffusers_version"]
            return (post_process(image_list), pipeline.config)
        finally:
            self.mutex.release()

    def get_pipeline(self, model_name: str, revision: str, custom_pipeline, torch_dtype):
        logging.debug(
            f"Loading {model_name} to device {self.device_id} - {torch.cuda.get_device_name(self.device_id)}"
        )
        # clear gpu cache
        torch.cuda.set_device(self.device_id)
        with torch.no_grad():
            torch.cuda.empty_cache()

        # load the pipeline and send it to the gpu
        pipeline = DiffusionPipeline.from_pretrained(
            model_name,
            use_auth_token=self.auth_token,
            device_map="auto",
            revision=revision,
            torch_dtype=torch_dtype,            
            custom_pipeline=custom_pipeline,
        )
        return pipeline.to(f"cuda:{self.device_id}")

    def log_device(self):
        logging.debug(
            f"Using device# {self.device_id} - {torch.cuda.get_device_name(self.device_id)}"
        )


def post_process(image_list) -> Image.Image:
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
    else:
        raise (Exception("too many images"))

    return image


def image_grid(image_list, rows, cols) -> Image.Image:
    w, h = image_list[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(image_list):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid
