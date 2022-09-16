import torch
import logging
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
import pickle


def preload_pipelines(model_name, auth_token, pipeline_names = ["txt2img", "img2img", "imginpaint"]):
    logging.debug(f"Using device# {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # this will preload all the pipelines and serialize them to disk
    # get_pipeline will then retrive from disk, this accomplishes two things:
    # 1 - pay the startup cost to get the model form hugging face only 1 time per process
    # 2 keep them out of RAM (main and GPU) until actually needed
    # on demand they get deserialized and pushed to the gpu
    #
    # TODO #8 model the GPU as a class; including what pipeline is loaded and if it has a workload or not
    # TODO #10 memory manage the pipelines so they don't all need to be in RAM at the same time (load, serialize, deserialize on demand)
    if "txt2img" in pipeline_names:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=auth_token,
        )

        pickle.dump(pipeline, open("/tmp/txt2img.pipeline", "wb"))

    if "img2img" in pipeline_names:
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_name,
            revision="fp16", 
            torch_dtype=torch.float16,
            use_auth_token=auth_token,
        )

        pickle.dump(pipeline, open("/tmp/img2img.pipeline", "wb"))

    if "imginpaint" in pipeline_names:
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_name,
            revision="fp16", 
            torch_dtype=torch.float16,
            use_auth_token=auth_token,
        )

        pickle.dump(pipeline, open("/tmp/imginpaint.pipeline", "wb"))


def get_pipeline(pipeline_name):
    with torch.no_grad():
        torch.cuda.empty_cache()

    pipe = pickle.load(open(f"/tmp/{pipeline_name}.pipeline", "rb"))
    return pipe.to("cuda")
