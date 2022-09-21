from .. import info
from urllib.parse import unquote
import logging
import io
import base64
from enum import auto
from fastapi_restful.enums import StrEnum


class format_enum(StrEnum):
    jpeg = auto()
    json = auto()
    png = auto()


def package_metadata(buffer, pipeline_config, args):
    metadata = info()
    metadata["pipeline_config"] = pipeline_config
    metadata["image"] = base64.b64encode(buffer.getvalue()).decode("UTF-8")
    metadata["parameters"] = args

    return metadata


def generate_buffer(device, **kwargs):
    try:
        logging.info(f"START generating {kwargs['pipeline_name']}")

        kwargs["prompt"] = clean_prompt(kwargs["prompt"])

        image, pipe_config = device(**kwargs)

        logging.info(f"END generating {kwargs['pipeline_name']}")
    except:
        raise Exception(423)

    format = (
        "JPEG" if kwargs["format"] == format_enum.json else kwargs["format"].upper()
    )
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)

    # we return kwargs so that it can be used as metadata if needed
    return buffer, pipe_config, kwargs


def clean_prompt(str):
    encoded = unquote(str).encode("utf8", "ignore")
    decoded = encoded.decode("utf8", "ignore")
    cleaned = decoded.replace('"', "").replace("'", "").strip()
    if len(cleaned) > 280:  # max length of a tweet
        raise Exception("prompt must be less than 281 characters")

    return cleaned
