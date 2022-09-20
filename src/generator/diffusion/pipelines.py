from typing import Dict, Any
import torch
import logging
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)
import pickle


class Pipelines:

    model_name: str = ""
    model_cache_dir: str = ""
    files: Dict[str, Any] = {}

    def __init__(self, model_name, model_cache_dir="/tmp") -> None:
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir

    def preload_pipelines(
        self,
        auth_token,
        pipeline_names=["txt2img", "img2img", "imginpaint"],
        conserve_memory=True,
    ):
        # this will preload all the pipelines and serialize them to disk.
        # the pre_load function then opens and keeps open a handle to each file to keep them locked
        # get_pipeline will then retrive from disk, accomplishing two things:
        # 1 - pay the startup cost to get the model form hugging face only 1 time per process
        # 2 - keep them out of RAM (main and GPU) until actually needed
        # on demand they get deserialized and pushed to the gpu
        #
        pipeline_type_map = {
            "txt2img": StableDiffusionPipeline,
            "img2img": StableDiffusionImg2ImgPipeline,
            "imginpaint": StableDiffusionInpaintPipeline,
        }

        for pipe_line_name in pipeline_names:
            StableDiffusionType = pipeline_type_map[pipe_line_name]
            if conserve_memory:
                pipeline = StableDiffusionType.from_pretrained(
                    self.model_name,
                    revision="fp16",
                    torch_dtype=torch.float16,
                    use_auth_token=auth_token,
                )
                pipeline.enable_attention_slicing()
            else:
                pipeline = StableDiffusionType.from_pretrained(
                    self.model_name,
                    use_auth_token=auth_token,
                )

            self.serialize_pipeline(
                pipeline, pipe_line_name, "fp16" if conserve_memory else "full"
            )

        return self

    def serialize_pipeline(self, pipeline, pipeline_name, revision):
        logging.debug(f"Serializing {pipeline_name}")

        model_name_path_part = self.model_name.replace("/", ".")
        pickle.dump(
            pipeline,
            open(
                f"/tmp/{model_name_path_part}.{revision}.{pipeline_name}.pipeline", "wb"
            ),
        )

        # open aand lock the file for later use
        self.files[pipeline_name] = open(
            f"/tmp/{model_name_path_part}.{revision}.{pipeline_name}.pipeline", "rb"
        )

    def load_pipeline(self, pipeline_name):
        logging.debug(
            f"Deserializing {pipeline_name} to device {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}"
        )
        # clear gpu memory
        with torch.no_grad():
            torch.cuda.empty_cache()

        # resurrect the new pipeline, send it to the device, and cache it in memory
        file = self.files[pipeline_name]
        pipe = pickle.load(file)
        file.seek(0, 0)  # set the file stream back to the beginning
        return pipe.to("cuda")
