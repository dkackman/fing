import requests
from PIL import Image
from io import BytesIO


def get_image(uri):
    response = requests.get(uri)
    image = Image.open(BytesIO(response.content))
    return image.resize((768, 512))
