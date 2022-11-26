from typing import List, Union
import json
import os
from pathlib import Path


def load_settings():
    with open(get_settings_full_path(), "r") as file:
        dict = json.loads(file.read())

    settings = Settings()
    settings.huggingface_token = dict.get("huggingface_token", True)
    settings.host = dict.get("host", "localhost")
    settings.port = dict.get("port", 9147)
    settings.log_level = dict.get("log_level", "DEBUG")
    settings.log_filename = dict.get("log_filename", "log/generator.log")
    settings.x_api_key_enabled = dict.get("x_api_key_enabled", False)
    settings.x_api_key_list = dict.get("x_api_key_list", [])
    settings.sdaas_token = dict.get("sdaas_token", "")

    # override settings file with environment vairables
    if not os.environ.get("FING_HOST", None) is None:
        settings.host = os.environ.get("FING_HOST", "")

    if not os.environ.get("HUGGINGFACE_TOKEN", None) is None:
        settings.huggingface_token = os.environ.get("HUGGINGFACE_TOKEN", "")

    if not os.environ.get("FING_X_API_KEY", None) is None:
        settings.x_api_key_enabled = True
        settings.x_api_key_list.append(os.environ.get("FING_X_API_KEY", ""))

    if not os.environ.get("SDAAS_TOKEN", None) is None:
        settings.huggingface_token = os.environ.get("SDAAS_TOKEN", "")

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


class Settings:
    # when true huggingface will look for auth from the environment - otherwise the api key itself
    huggingface_token: Union[bool, str] = True
    host: str = "localhost"
    port: int = 9147
    log_level: str = "DEBUG"
    log_filename: str = "log/generator.log"
    x_api_key_enabled: bool = False
    x_api_key_list: List[str] = []
    sdaas_token: str = ""
