from typing import List, Union
from pydantic import BaseSettings
import json
import os
from pathlib import Path


def load_settings():
    with open(get_settings_full_path(), "r") as file:
        dict = json.loads(file.read())
    settings = Settings(**dict)

    # override settings file with environment vairables (mainly for docker)
    if not os.environ.get("FING_HOST", None) is None:
        settings.host = os.environ.get("FING_HOST", '')

    if not os.environ.get("HUGGINGFACE_TOKEN", None) is None:
        settings.huggingface_token = os.environ.get("HUGGINGFACE_TOKEN", '')

    if not os.environ.get("MODEL_CACHE_DIR", None) is None:
        settings.model_cache_dir = os.environ.get("MODEL_CACHE_DIR", '')

    if not os.environ.get("FING_X_API_KEY", None) is None:
        settings.x_api_key_enabled = True
        settings.x_api_key_list.append(os.environ.get("FING_X_API_KEY", ''))

    return settings


def save_settings(settings):
    dir = Path(get_settings_dir())
    dir.mkdir(0o770, parents=True, exist_ok=True)

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
    # when true huggingface will look for auth from the environment - otherwise the api key itself
    huggingface_token: Union[bool, str] = True
    host: str = "localhost"
    port: int = 9147
    log_level: str = "DEBUG"
    log_filename: str = "log/generator.log"
    model_cache_dir: str = "/tmp"
    x_api_key_enabled: bool = False
    x_api_key_list: List[str] = []
    conserve_memory: bool = True  # deprecated
