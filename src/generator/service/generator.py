from ..diffusion.device import Device
from .. import info, Software
from urllib.parse import unquote
import logging
import io
import base64
from enum import auto
from fastapi import HTTPException
from fastapi_restful.enums import StrEnum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from PIL import Image
import requests


class image_format_enum(StrEnum):
    jpeg = auto()
    json = auto()
    png = auto()


class audio_format_enum(StrEnum):
    wav = auto()
    json = auto()


class PipelineConfig(BaseModel):
    vae: Optional[List[str]]
    text_encoder: Optional[List[str]]
    tokenizer: Optional[List[str]]
    unet: List[str]
    scheduler: List[str]
    safety_checker: Optional[List[str]]
    feature_extractor: Optional[List[str]]
    seed: Optional[int]


class PipelineConfigModel(BaseModel):
    pipeline_config: PipelineConfig


class PackageMetaDataModel(BaseModel):
    pipeline_config: Dict[str, Any]
    software: Software
    image: str
    parameters: Dict[str, Any]


def package_metadata(buffer, pipeline_config, args) -> PackageMetaDataModel:
    software = info().software
    #pipeline_config = PipelineConfig.parse_obj(pipeline_config)
    image = base64.b64encode(buffer.getvalue()).decode("UTF-8")

    # torch_dtype is not iterable and won't serialize
    # filter out any images from the metadata
    serlized_args = {
        k: v
        for (k, v) in args.items()
        if not isinstance(v, Image.Image) and not k == "torch_dtype"
    }

    return PackageMetaDataModel(
        pipeline_config=pipeline_config,
        software=software,
        image=image,
        parameters=serlized_args,
    )


def generate_buffer(device: Device, **kwargs):
    format = kwargs.pop("format", "JPEG").upper()
    format = format if format != "JSON" else "JPEG"
    pipeline_name = kwargs.pop("pipeline_name", "unknown pipeline")
    try:
        logging.info(
            f"START generating {pipeline_name} on device {device.device_id}"
        )

        if "prompt" in kwargs:
            kwargs["prompt"] = clean_prompt(kwargs["prompt"])

        image, pipe_config = device(**kwargs)  # type: ignore

        logging.info(
            f"END generating {pipeline_name} on device {device.device_id}"
        )
    except Exception as e:
        print(e)

        if len(e.args) > 0:
            if e.args[0] == "busy":
                raise HTTPException(423)
            if e.args[0] == "NSFW":
                raise HTTPException(406)  # Not Acceptable

        raise HTTPException(500)

    buffer = image_to_buffer(image, format)

    # we return kwargs so that it can be used as metadata if needed
    return buffer, pipe_config, kwargs


def get_image(uri):
    response = requests.get(uri)
    image = Image.open(io.BytesIO(response.content))
    maxzise = 1024
    if image.height > maxzise or image.width > maxzise:
        image.thumbnail((maxzise, maxzise), Image.Resampling.LANCZOS)

    return image


def image_to_buffer(image, format):
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)

    return buffer


def clean_prompt(str):
    encoded = unquote(str).encode("utf8", "ignore")
    decoded = encoded.decode("utf8", "ignore")
    cleaned = decoded.replace('"', "").replace("'", "").strip()

    return cleaned
