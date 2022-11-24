from typing import Optional
import torch
import logging
from PIL import Image
from threading import Lock
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler


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
            # convert the name of this api parameter to the diffuser equivalent (name changed)
            kwargs["num_images_per_prompt"] = kwargs.pop("num_images", 1)

            if "prompt" in kwargs:
                logging.info(f"Prompt is {kwargs['prompt']}")
            self.log_device()

            # if no scheduler is provided default to DPMSolverMultistepScheduler
            scheduler = kwargs.pop("scheduler", None)
            model_name = kwargs.pop("model_name")
            if scheduler is None:
                scheduler = DPMSolverMultistepScheduler.from_pretrained(
                    model_name,
                    use_auth_token=self.auth_token,
                    subfolder="scheduler",
                )

            pipeline = self.get_pipeline(
                model_name,
                kwargs.pop("revision"),
                kwargs.pop("custom_pipeline", None),
                kwargs.pop("torch_dtype", torch.float16),
                scheduler,
                kwargs.pop("pipeline_type", DiffusionPipeline),
            )

            # this allows reproducability
            seed: Optional[int] = kwargs.pop("seed", None)
            if seed is None:
                seed = torch.seed()
            torch.manual_seed(seed)

            p = pipeline(**kwargs)

            # if only one image (the usual case) and nsfw raise exception
            if (
                hasattr(p, "nsfw_content_detected")
                and len(p.nsfw_content_detected) == 1
            ):
                for _ in filter(lambda nsfw: nsfw, p.nsfw_content_detected):
                    raise Exception("NSFW")

            pipeline.config["seed"] = seed
            pipeline.config["class_name"] = pipeline.config["_class_name"]
            pipeline.config["diffusers_version"] = pipeline.config["_diffusers_version"]

            return (post_process(p.images), pipeline.config)
        finally:
            self.mutex.release()

    def get_pipeline(
        self,
        model_name: str,
        revision: str,
        custom_pipeline,
        torch_dtype,
        scheduler,
        pipeline_type,
    ):
        logging.debug(
            f"Loading {model_name} to device {self.device_id} - {torch.cuda.get_device_name(self.device_id)}"
        )
        # clear gpu cache
        torch.cuda.set_device(self.device_id)
        with torch.no_grad():
            torch.cuda.empty_cache()

        # load the pipeline and send it to the gpu
        pipeline = pipeline_type.from_pretrained(
            model_name,
            use_auth_token=self.auth_token,
            revision=revision,
            torch_dtype=torch_dtype,
            custom_pipeline=custom_pipeline,
            scheduler=scheduler,
        )
        return pipeline.to(f"cuda:{self.device_id}")  # type: ignore

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
