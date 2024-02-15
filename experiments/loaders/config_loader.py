import os
import yaml

from typing import Any, Dict


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, mode="r") as file:
        config = yaml.safe_load(file.read())
    return config
