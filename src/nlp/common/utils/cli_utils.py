import logging
from pathlib import Path
from typing import Any

from nlp.common.utils.file.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_cli_config(config_file_path: str | Path | None = None, **kwargs: Any) -> dict[str, Any]:
    """
    Load configuration from a file and merge it with runtime arguments.
    """
    if config_file_path:
        logger.info(f'Loading configuration from {config_file_path}')
        merged = load_config(config_file_path)
        merged.update(kwargs)
        logger.info(f'Merged config from file with overrides: {kwargs.keys() if kwargs else {}}')
    else:
        merged = kwargs
        logger.info('No config file provided; using runtime arguments only.')
    return merged
