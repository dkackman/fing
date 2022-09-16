import torch
import logging
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline


pipelines = {}

def preload_pipelines(model_name, auth_token, pipeline_names = ["txt2img", "img2img", "imginpaint"]):
    logging.debug(f"Using device# {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # this will preload the pieline into CPU memory
    # on demand they get swapped into gpu memory
    #
    # TODO #8 model the GPU as a class; including what pipeline is loaded and if it has a workload or not
    # TODO #9 implement img in-painting
    # TODO #10 memory manage the pipelines so they don't all need to be in RAM at the same time (load, serialize, deserialize on demand)
    if "txt2img" in pipeline_names:
        pipelines["txt2img"] = StableDiffusionPipeline.from_pretrained(
            model_name,
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=auth_token,
        )

    if "img2img" in pipeline_names:
        pipelines["img2img"] = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_name,
            revision="fp16", 
            torch_dtype=torch.float16,
            use_auth_token=auth_token,
        )

    if "imginpaint" in pipeline_names:
        pipelines["imginpaint"] = StableDiffusionInpaintPipeline.from_pretrained(
            model_name,
            revision="fp16", 
            torch_dtype=torch.float16,
            use_auth_token=auth_token,
        )


def get_pipeline(pipeline_name):
    with torch.no_grad():
        torch.cuda.empty_cache()

    return pipelines[pipeline_name].to("cuda")
