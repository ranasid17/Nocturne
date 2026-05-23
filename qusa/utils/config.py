# qusa/utils/config.py

import os
import yaml

from pathlib import Path


def load_env(env_path=".env"):
    """
    Load environment variables from a .env file.
    Only basic KEY=VALUE pairs are supported.
    """
    env_path = Path(env_path).expanduser()
    if not env_path.exists():
        return

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.

    Parameters:
        1) config_path (str): Path to the YAML configuration file.
    """

    config_path = Path(config_path).expanduser()

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # expand user paths
    for section in config.values():
        if isinstance(section, dict):
            for key, value in section.items():
                if isinstance(value, str) and value.startswith("~"):
                    section[key] = os.path.expanduser(value)
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, str) and subvalue.startswith("~"):
                            section[key][subkey] = os.path.expanduser(subvalue)
    return config
