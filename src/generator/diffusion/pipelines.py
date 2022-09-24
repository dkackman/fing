from typing import Dict, Any
import torch
import logging
import pickle


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
        conserve_memory=True,
    ):
        # this will preload all the pipelines and serialize them to disk.
        # the pre_load function then opens and keeps open a handle to each file to keep them locked
        # get_pipeline will then retrive from disk, accomplishing two things:
        # 1 - pay the startup cost to get the model form hugging face only 1 time per process
        # 2 - keep them out of RAM (main and GPU) until actually needed
        # on demand they get deserialized and pushed to the gpu

        for pipeline_name, StableDiffusionType in pipeline_type_map.items():
            if conserve_memory:
                pipeline = StableDiffusionType.from_pretrained(
                    model_name,
                    revision="fp16",
                    torch_dtype=torch.float16,
                    use_auth_token=auth_token,
                )
                pipeline.enable_attention_slicing()
            else:
                pipeline = StableDiffusionType.from_pretrained(
                    model_name,
                    use_auth_token=auth_token,
                )

            self.serialize_pipeline(
                pipeline,
                model_name,
                pipeline_name,
                "fp16" if conserve_memory else "full",
            )

        return self

    def serialize_pipeline(self, pipeline, model_name, pipeline_name, revision):
        logging.debug(f"Serializing {pipeline_name}")

        model_name_path_part = model_name.replace("/", ".")
        pickle.dump(
            pipeline,
            open(
                f"/tmp/{model_name_path_part}.{revision}.{pipeline_name}.pipeline", "wb"
            ),
        )

        pipeline_key = f"{model_name}.{pipeline_name}"
        # open aand lock the file for later use
        self.files[pipeline_key] = open(
            f"/tmp/{model_name_path_part}.{revision}.{pipeline_name}.pipeline", "rb"
        )

    def load_pipeline(self, pipeline_key):
        # resurrect the requested pipeline
        file = self.files[pipeline_key]
        pipeline = pickle.load(file)
        file.seek(0, 0)  # set the file stream back to the beginning
        return pipeline
