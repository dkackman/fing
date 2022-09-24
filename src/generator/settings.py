from operator import truediv
from typing import List
from pydantic import BaseSettings
import json
import os
from pathlib import Path


def load_settings():
    with open(get_settings_full_path(), "r") as file:
        dict = json.loads(file.read())
    return Settings(**dict)


def save_settings(settings):
    with open(get_settings_full_path(), "w") as file:
        file.write(settings.json(indent=2))


def settings_exist():
    return get_settings_full_path().is_file()


def resolve_path(path):
    path = get_settings_dir().joinpath(path)
    # make the directory if it doesn't exist
    path.parent.mkdir(0o770, parents=True, exist_ok=True)

    return path


def get_settings_dir():
    dir = os.environ.get("FING_ROOT", None)
    if dir is None:
        dir = "~/.fing/"

    return Path(dir).expanduser()


def get_settings_full_path():
    return Path(get_settings_dir()).joinpath("settings.json")


class Settings(BaseSettings):
    huggingface_token: str = "PLACE YOUR TOKEN HERE"
    host: str = "localhost"
    port: int = 9147
    log_level: str = "DEBUG"
    log_filename: str = "log/generator.log"
    model_cache_dir: str = "/tmp"
    x_api_key_enabled: bool = False
    x_api_key_list: List[str] = []
    conserve_memory: bool = True
