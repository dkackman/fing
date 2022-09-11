import os
from tkinter import N
import yaml
from pathlib import Path

class Config:
    config_file = None
    root_dir = None
    def __init__(self) -> None:
        dir = os.environ.get("GENERATOR_ROOT", None)
        if dir is None:
            dir = "~/.fing/"

        self.root_dir = Path(dir).expanduser()
        if not self.root_dir.exists():
            raise Exception(f"config dir not found {dir}")

    def load(self):
        filename = self.root_dir.joinpath("config.yaml")
        if not filename.is_file():
            raise Exception("config file not found")

        with open(filename, "r") as file:
            self.config_file = yaml.safe_load(file)

        if self.config_file is None:
            raise Exception("could not open config file")

        return self

    def resolve_path(self, path):
        path = self.root_dir.joinpath(path)
        # mkae the directory if it doesn't exist
        path.parent.mkdir(0o770, parents=True, exist_ok=True)

        return path
