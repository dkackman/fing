import torch
import logging
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
import pickle

revision = "fp16"
torch_dtype=torch.float16

def preload_pipelines(model, auth_token, pipeline_names = ["txt2img", "img2img", "imginpaint"]):
    # this will preload all the pipelines and serialize them to disk
    # get_pipeline will then retrive from disk, accomplishing two things:
    # 1 - pay the startup cost to get the model form hugging face only 1 time per process
    # 2 - keep them out of RAM (main and GPU) until actually needed
    # on demand they get deserialized and pushed to the gpu
    
    global model_name #TODO - make this a class
    model_name = model.replace("/", ".")

    if "txt2img" in pipeline_names:
        logging.debug("Loading txt2img")
        pipeline = StableDiffusionPipeline.from_pretrained(
            model,
            revision=revision,
            torch_dtype=torch_dtype,
            use_auth_token=auth_token,
        )

        pickle.dump(pipeline, open(f"/tmp/{model_name}.{revision}.txt2img.pipeline", "wb"))

    if "img2img" in pipeline_names:
        logging.debug("Loading img2img")
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            model,
            revision=revision, 
            torch_dtype=torch_dtype,
            use_auth_token=auth_token,
        )

        pickle.dump(pipeline, open(f"/tmp/{model_name}.{revision}.img2img.pipeline", "wb"))

    if "imginpaint" in pipeline_names:
        logging.debug("Loading imginpaint")
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model,
            revision=revision, 
            torch_dtype=torch_dtype,
            use_auth_token=auth_token,
        )

        pickle.dump(pipeline, open(f"/tmp/{model_name}.{revision}.imginpaint.pipeline", "wb"))


def get_pipeline(pipeline_name):
    logging.debug(f"Deserializeg {pipeline_name} to {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}") 

    with torch.no_grad():
        torch.cuda.empty_cache()

    pipe = pickle.load(open(f"/tmp/{model_name}.{revision}.{pipeline_name}.pipeline", "rb"))
    return pipe.to("cuda")
