from typing import Dict, Type, Union
import logging
import pickle
from pathlib import Path
import io


class PipelineCache:

    pipeline_cache_dir: str = ""
    files: Dict[str, Path]

    def __init__(self, pipeline_cache_dir: str = "/tmp") -> None:
        self.pipeline_cache_dir = pipeline_cache_dir
        self.files = dict[str, Path]()

    def preload(
        self,
        auth_token: Union[bool, str],
        model_name: str,
        pipeline_type_map: Dict[str, Type],
        enable_attention_slicing: bool = True,
    ):
        # this will preload all the pipelines and serialize them to disk.
        # the pre_load function then opens and keeps open a handle to each file to keep them locked
        # get_pipeline will then retrieve thm from disk, accomplishing two things:
        # 1 - pay the startup cost to get the model from hugging face only 1 time per process
        # 2 - keep them out of RAM (main and GPU) until actually needed
        # on demand they get deserialized and pushed to the gpu

        for pipeline_name, StableDiffusionType in pipeline_type_map.items():
            # if the model isn't cached go load it
            filepath = self.get_pipeline_filepath(model_name, pipeline_name)
            if not Path(filepath).is_file():
                print(f"Loading {model_name}.{pipeline_name}")
                pipeline = StableDiffusionType.from_pretrained(
                    model_name,
                    use_auth_token=auth_token,
                )
                if enable_attention_slicing:
                    pipeline.enable_attention_slicing()

                print(f"Serializing to {filepath}")
                self.serialize_pipeline(
                    pipeline,
                    model_name,
                    pipeline_name,
                )

            pipeline_key = f"{model_name}.{pipeline_name}"
            # open and lock the file for later use
            self.files[pipeline_key] = filepath #open(filepath, "rb")

        return self

    def serialize_pipeline(self, pipeline, model_name: str, pipeline_name: str):
        logging.debug(f"Serializing {pipeline_name}")

        filepath = self.get_pipeline_filepath(model_name, pipeline_name)
        pickle.dump(
            pipeline,
            open(filepath, "wb"),
        )

    def load_pipeline(self, model_name: str, pipeline_name: str):
        pipeline_key = f"{model_name}.{pipeline_name}"

        # resurrect the requested pipeline
        with open(self.files[pipeline_key], "rb") as file:
            pipeline = pickle.load(file)

        # this will be on the cpu device - up to caller to move it
        return pipeline

    def get_pipeline_filepath(self, model_name: str, pipeline_name: str) -> Path:
        model_name_path_part = model_name.replace("/", ".")
        pipeline_path_part = f"{model_name_path_part}.{pipeline_name}.pipeline"
        return Path(self.pipeline_cache_dir).joinpath(pipeline_path_part)
