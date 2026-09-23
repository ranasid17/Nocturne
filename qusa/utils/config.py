# qusa/utils/config.py

from qusa.utils.settings import PROJECT_ROOT, load_settings


def load_env(env_path=".env"):
    """
    Load environment variables from a .env file.
    Only basic KEY=VALUE pairs are supported.
    """
    from dotenv import load_dotenv

    load_dotenv(env_path, override=False)


def load_config(config_path=None):
    """
    Load configuration from a YAML file.

    Parameters:
        1) config_path (str): Path to the YAML configuration file.
    """

    return load_settings(config_path, project_root=PROJECT_ROOT)
