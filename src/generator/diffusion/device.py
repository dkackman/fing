import torch
import logging
from torch.cuda.amp import autocast
from PIL import Image


class Device:
    pipelines = None

    def __init__(self, pipelines) -> None:
        self.pipelines = pipelines

    def __call__(self, **kwargs):
        num_images = kwargs["num_images"] if "num_images" in kwargs else 1
        if num_images > 4:
            raise Exception("The maximum number of images is 4")

        logging.info(f"Prompt is {kwargs['prompt']}")
        log_device()

        pipeline = self.pipelines.load_pipeline(kwargs["pipeline_name"])
        image_list = []
        # this can be done in a single pass to the pipeline but consumes a lot of memory and isn't much faster
        for i in range(num_images):
            with autocast():
                # this comprehension expresssion strips items from kwargs that aren't recognized by the pipeline
                image = pipeline(
                    **{
                        key: value
                        for key, value in kwargs.items()
                        if key != "pipeline_name" and key != "num_images"
                    }
                ).images[0]
                image_list.append(image)

        return (post_process(image_list), pipeline.config)


def log_device():
    logging.debug(
        f"Using device# {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}"
    )


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
