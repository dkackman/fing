from .. import info, Software
from urllib.parse import unquote
import logging
import io
import base64
from enum import auto
from fastapi_restful.enums import StrEnum
from typing import Any, Dict, List
from pydantic import BaseModel


class format_enum(StrEnum):
    jpeg = auto()
    json = auto()
    png = auto()


class PipelineConfig(BaseModel):
    vae: List[str]
    _class_name: str
    _diffusers_version: str
    text_encoder: List[str]
    tokenizer: List[str]
    unet: List[str]
    scheduler: List[str]
    safety_checker: List[str]
    feature_extractor: List[str]


class PipelineConfigModel(BaseModel):
    pipeline_config: PipelineConfig


class PackageMetaDataModel(BaseModel):
    pipeline_config: PipelineConfig
    software: Software
    image: str
    parameters: Dict[str, Any]


def package_metadata(buffer, pipeline_config, args) -> PackageMetaDataModel:
    software = info().software
    pipeline_config = PipelineConfig.parse_obj(pipeline_config)
    image = base64.b64encode(buffer.getvalue()).decode("UTF-8")

    # dyanmically typed version in case this is too brittle
    # metaddata = info().dict
    # metadata["pipeline_config"] = pipeline_config
    # metadata["image"] = base64.b64encode(buffer.getvalue()).decode("UTF-8")
    # metadata["parameters"] = args

    return PackageMetaDataModel(
        pipeline_config=pipeline_config, software=software, image=image, parameters=args
    )


def generate_buffer(device, **kwargs):
    format = kwargs.pop("format", "JPEG").upper()

    try:
        logging.info(f"START generating {kwargs['pipeline_name']}")

        if "prompt" in kwargs:
            kwargs["prompt"] = clean_prompt(kwargs["prompt"])

        image, pipe_config = device(**kwargs)

        logging.info(f"END generating {kwargs['pipeline_name']}")
    except:
        raise Exception(423)

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
