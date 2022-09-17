from token import EXACT_TOKEN_TYPES
import torch
import logging
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
import pickle


class Pipelines:

    revision = None
    torch_dtype = None
    model_name = None
    files = {}
    last_pipe = None    # this only works for single gpu implementation right now
                        # TODO #12 cache the last pipeline per device

    def __init__(self, model_name, revision="fp16", torch_dtype=torch.float16) -> None:
        self.revision = revision
        self.torch_dtype = torch_dtype
        self.model_name = model_name


    def preload_pipelines(self, auth_token, pipeline_names = ["txt2img", "img2img", "imginpaint"]):
        # this will preload all the pipelines and serialize them to disk. 
        # the pre_load function then opens and keeps open a handle to each file to keep them locked
        # get_pipeline will then retrive from disk, accomplishing two things:
        # 1 - pay the startup cost to get the model form hugging face only 1 time per process
        # 2 - keep them out of RAM (main and GPU) until actually needed
        # on demand they get deserialized and pushed to the gpu        

        model_name_path_part = self.model_name.replace("/", ".")
        if "txt2img" in pipeline_names:
            logging.debug("Loading txt2img")
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_name,
                revision=self.revision,
                torch_dtype=self.torch_dtype,
                use_auth_token=auth_token,
            )

            pickle.dump(pipeline, open(f"/tmp/{model_name_path_part}.{self.revision}.txt2img.pipeline", "wb"))
            self.files["txt2img"] = open(f"/tmp/{model_name_path_part}.{self.revision}.txt2img.pipeline", "rb")

        if "img2img" in pipeline_names:
            logging.debug("Loading img2img")
            pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_name,
                revision=self.revision, 
                torch_dtype=self.torch_dtype,
                use_auth_token=auth_token,
            )

            pickle.dump(pipeline, open(f"/tmp/{model_name_path_part}.{self.revision}.img2img.pipeline", "wb"))
            self.files["img2img"] = open(f"/tmp/{model_name_path_part}.{self.revision}.img2img.pipeline", "rb")

        if "imginpaint" in pipeline_names:
            logging.debug("Loading imginpaint")
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_name,
                revision=self.revision, 
                torch_dtype=self.torch_dtype,
                use_auth_token=auth_token,
            )

            pickle.dump(pipeline, open(f"/tmp/{model_name_path_part}.{self.revision}.imginpaint.pipeline", "wb"))
            self.files["imginpaint"] = open(f"/tmp/{model_name_path_part}.{self.revision}.imginpaint.pipeline", "rb")


    def get_pipeline(self, pipeline_name):
        # if the last pipeline is the one requested, just return it
        if self.last_pipe is not None:
            if self.last_pipe[0] == pipeline_name:
                logging.debug(f"{pipeline_name} already loaded") 
                return self.last_pipe[1]
            else:
                del self.last_pipe      # if there is a loaded pipeline but it's different clean up the memory

        if torch.cuda.is_available():
            logging.debug(f"Deserializing {pipeline_name} to device {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}") 
            # clear gpu memory
            with torch.no_grad():
                torch.cuda.empty_cache()
        else:
            logging.debug(f"Deserializing {pipeline_name} to cpu") 

        # resurrect the new pipeline, send it to the device, and cache it in memory
        file = self.files[pipeline_name]
        pipe = pickle.load(file)
        file.seek(0, 0) # set the file stream back to the beginning
        the_pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        self.last_pipe = (pipeline_name, the_pipe)

        return the_pipe
