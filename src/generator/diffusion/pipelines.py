from typing import Dict, Any
from xmlrpc.client import boolean
import torch
import logging
import pickle
from pathlib import Path


class Pipelines:

    model_cache_dir: str = ""
    files: Dict[str, Any] = {}

    def __init__(self, model_cache_dir="/tmp") -> None:
        self.model_cache_dir = model_cache_dir

    def preload_pipelines(
        self,
        auth_token: str,
        model_name: str,
        pipeline_type_map,
        revision: str = "main",
        torch_dtype=torch.float16,
        enable_attention_slicing: boolean = True,
    ):
        # this will preload all the pipelines and serialize them to disk.
        # the pre_load function then opens and keeps open a handle to each file to keep them locked
        # get_pipeline will then retrive from disk, accomplishing two things:
        # 1 - pay the startup cost to get the model form hugging face only 1 time per process
        # 2 - keep them out of RAM (main and GPU) until actually needed
        # on demand they get deserialized and pushed to the gpu

        for pipeline_name, StableDiffusionType in pipeline_type_map.items():
            # if the model isn't cached go load it
            filepath = get_pipeline_filepath(model_name, pipeline_name, revision)
            if not Path(filepath).is_file():
                pipeline = StableDiffusionType.from_pretrained(
                    model_name,
                    revision=revision,
                    torch_dtype=torch_dtype,
                    use_auth_token=auth_token,
                )
                if enable_attention_slicing:
                    pipeline.enable_attention_slicing()

                self.serialize_pipeline(
                    pipeline,
                    model_name,
                    pipeline_name,
                    revision,
                )

            pipeline_key = f"{model_name}.{pipeline_name}"
            # open aand lock the file for later use
            self.files[pipeline_key] = open(filepath, "rb")

        return self

    def serialize_pipeline(
        self, pipeline, model_name: str, pipeline_name: str, revision: str
    ):
        logging.debug(f"Serializing {pipeline_name}")

        filepath = get_pipeline_filepath(model_name, pipeline_name, revision)
        pickle.dump(
            pipeline,
            open(filepath, "wb"),
        )

    def load_pipeline(self, pipeline_key: str):
        # resurrect the requested pipeline
        file = self.files[pipeline_key]
        pipeline = pickle.load(file)
        file.seek(0, 0)  # set the file stream back to the beginning
        return pipeline


def get_pipeline_filepath(model_name: str, pipeline_name: str, revision: str) -> str:
    model_name_path_part = model_name.replace("/", ".")
    return f"/tmp/{model_name_path_part}.{revision}.{pipeline_name}.pipeline"
