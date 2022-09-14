import requests
from PIL import Image
from io import BytesIO
import logging

def get_image(uri):
    logging.debug(f"Downloadinf {uri}")
    response = requests.get(uri)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    return init_image.resize((768, 512))
