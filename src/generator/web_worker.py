from flask_restful import abort
import logging
from urllib.parse import unquote
import io
from threading import Lock
import torch

# TODO #5 create a worker per GPU

mutex = Lock()

def generate_txt2img_buffer(model, guidance_scale, num_inference_steps, num_images, height, width, prompt):
    try:
        # only allow one image generation at a time        
        locked = mutex.acquire(False)
        if locked:
            logging.info(f"START txt2img generating")

            image, config = model.get_txt2img( 
                guidance_scale, 
                num_inference_steps, 
                num_images, 
                height, 
                width, 
                prompt
            )
            logging.info(f"END txt2img generating")
        else:
            abort(423, "Busy. Try again later.")
    finally:
        if locked:
            mutex.release()

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    return buffer, config


def generate_img2img_buffer(model, strength, guidance_scale, num_inference_steps, num_images, prompt, init_image):
    try:
        # only allow one image generation at a time        
        locked = mutex.acquire(False)
        if locked:
            logging.info(f"START img2img generating")

            image, config = model.get_img2img( 
                strength,
                guidance_scale, 
                num_inference_steps, 
                num_images, 
                prompt,
                init_image
            )
            logging.info(f"END img2img generating")
        else:
            abort(423, "Busy. Try again later.")
    finally:
        if locked:
            mutex.release()

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    return buffer, config

def info():
    return {
            'software': {
                'name': "fing",
                'version': "0.1.0",
                'torch_version': torch.__version__,
            }
        }


# clean up the string - removing non utf-8 characters, check length
def clean_prompt(str):
    encoded = unquote(str).encode("utf8", "ignore")
    decoded = encoded.decode("utf8", "ignore")  
    cleaned = decoded.replace('"' , "").replace("'", "").strip()
    if len(cleaned) > 280: # max length of a tweet
        raise Exception("prompt must be less than 281 characters")
        
    return cleaned
