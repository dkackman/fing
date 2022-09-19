import os
import yaml
from pathlib import Path


class Config:
    config_file = None

    def __init__(self) -> None:
        if not Config.get_root_dir().exists():
            raise Exception(f"config dir not found {dir}")


    def load(self):
        filename = Config.get_full_path()
        if not filename.is_file():
            raise Exception("config file not found")

        with open(filename, "r") as file:
            self.config_file = yaml.safe_load(file)

        if self.config_file is None:
            raise Exception("could not open config file")

        return self


    def get_root_dir():
        dir = os.environ.get("FING_ROOT", None)
        if dir is None:
            dir = "~/.fing/"

        return Path(dir).expanduser()


    def get_full_path():
        return Path(Config.get_root_dir()).joinpath("config.yaml")


    def exists():
        return Config.get_root_dir().joinpath("config.yaml").is_file()


    def save_config(config):
        path = Config.get_root_dir()
        if not path.exists():
            path.parent.mkdir(0o770, parents=True, exist_ok=True)

        with open(path.joinpath("config.yaml"), "w") as f:
            yaml.safe_dump(config, f)


    def load_from(file_path):
        if not file_path.is_file():
            raise Exception("config file not found")

        with open(file_path, "r") as file:
            return yaml.safe_load(file)


    def resolve_path(path):
        path =  Config.get_root_dir().joinpath(path)
        # make the directory if it doesn't exist
        path.parent.mkdir(0o770, parents=True, exist_ok=True)

        return path
