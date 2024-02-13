import os
import yaml

from typing import Any, Dict


def load_config() -> Dict[str, Any]:
    with open(os.path.abspath("config.yaml"), mode="r") as file:
        config = yaml.safe_load(file.read())
    return config
