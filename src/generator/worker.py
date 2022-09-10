import torch
import logging
from diffusers import StableDiffusionPipeline
from torch.cuda.amp import autocast

def load_model(model, auth_token):
    logging.debug(f"Using device# {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=auth_token,
    )

    pipe.to("cuda")  # Run on GPU

    return pipe


def generate_with_pipe(pipe, guidance_scale, raw_prompt):
    prompt = clean(raw_prompt)
    logging.info(f"Prompt is [{prompt}]")
    logging.debug(f"Using device# {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")

    with autocast():
        image = pipe(prompt, guidance_scale=guidance_scale)["sample"][0]

    return image


# this does the actual image generation
def generate(model, guidance_scale, raw_prompt, auth_token):
    prompt = clean(raw_prompt)
    logging.info(f"Prompt is [{prompt}]")

    pipe = load_model(model, auth_token)

    with autocast():
        image = pipe(prompt, guidance_scale=guidance_scale)["sample"][0]

    return image

# clean up the string - removing non utf-8 characters, check length
def clean(str):
    encoded = str.encode("utf8", "ignore")
    decoded = encoded.decode("utf8", "ignore")
    cleaned = decoded.strip()
    if len(cleaned) > 280: # max length of a tweet
        raise Exception("prompt must be less than 281 characters")
        
    return cleaned
