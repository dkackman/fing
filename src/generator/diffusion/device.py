import torch
import logging
from torch.cuda.amp import autocast
from PIL import Image
from collections import namedtuple
from threading import Lock


pipeline_reference = namedtuple("pipeline_reference", ("key", "pipeline"))


class Device:
    device_id = None
    pipelines = None
    last_pipeline = None
    mutex = Lock()

    def __init__(self, device_id, pipelines) -> None:
        self.device_id = device_id
        self.pipelines = pipelines

    def __call__(self, **kwargs):
        if not self.mutex.acquire(False):
            raise (Exception("busy"))

        try:
            num_images = kwargs["num_images"] if "num_images" in kwargs else 1
            if num_images > 4:
                raise Exception("The maximum number of images is 4")

            logging.info(f"Prompt is {kwargs['prompt']}")
            self.log_device()

            pipeline = self.get_pipeline(kwargs["model_name"], kwargs["pipeline_name"])
            image_list = []
            # this can be done in a single pass to the pipeline but consumes a lot of memory and isn't much faster
            for i in range(num_images):
                with autocast():
                    # this comprehension expresssion strips items from kwargs that aren't recognized by the pipeline
                    image = pipeline(
                        **{
                            key: value
                            for key, value in kwargs.items()
                            if key != "pipeline_name"
                            and key != "model_name"
                            and key != "num_images"
                            and key != "format"
                        }
                    ).images[0]
                    image_list.append(image)

            return (post_process(image_list), pipeline.config)
        finally:
            self.mutex.release()

    def get_pipeline(self, model_name, pipeline_name):
        pipeline_key = f"{model_name}.{pipeline_name}"
        # if the last pipeline is the one requested, just return it
        if self.last_pipeline is not None:
            if self.last_pipeline.key == pipeline_key:
                logging.debug(f"{pipeline_key} already loaded")
                return self.last_pipeline.pipeline

            # if there is a loaded pipeline but it's different clean up the memory
            del self.last_pipeline

        logging.debug(
            f"Deserializing {pipeline_key} to device {self.device_id} - {torch.cuda.get_device_name(self.device_id)}"
        )
        # clear gpu cache
        torch.cuda.set_device(self.device_id)
        with torch.no_grad():
            torch.cuda.empty_cache()

        # get the cached pipeline and send it to the gpu
        new_pipeline = self.pipelines.load_pipeline(pipeline_key)
        gpu_pipeline = new_pipeline.to(f"cuda:{self.device_id}")
        # then delete the one in main memory right away since it is quite large
        del new_pipeline
        # and keep a reference to the one in the gpu
        self.last_pipeline = pipeline_reference(pipeline_key, gpu_pipeline)
        return gpu_pipeline

    def log_device(self):
        logging.debug(
            f"Using device# {self.device_id} - {torch.cuda.get_device_name(self.device_id)}"
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
