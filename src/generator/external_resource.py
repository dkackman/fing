import requests
from PIL import Image
from io import BytesIO
import logging


def get_image(uri):
    logging.debug(f"Downloading {uri}")
    response = requests.get(uri)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image.resize((768, 512))
