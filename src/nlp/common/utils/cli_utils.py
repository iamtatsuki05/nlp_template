import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from nlp.common.utils.file.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_cli_config(config_file_path: str | Path | None = None, **kwargs: object) -> dict[str, Any]:
    """Load configuration from a file and merge it with runtime arguments."""
    if config_file_path:
        logger.info('Loading configuration from %s', config_file_path)
        merged = load_config(config_file_path)
        merged.update(kwargs)
        logger.info('Merged config with overrides: %s', list(kwargs) if kwargs else [])
    else:
        merged = dict(kwargs)
        logger.info('No config file provided; using runtime arguments only.')
    return merged


def load_and_parse_config[T: BaseModel](
    config_class: type[T],
    config_file_path: str | Path | None = None,
    **kwargs: object,
) -> T:
    """Load configuration from file, merge with kwargs, and parse into Pydantic model.

    This function provides type-safe configuration loading by:
    1. Loading config from file (if provided)
    2. Merging with CLI overrides
    3. Validating and parsing into the specified Pydantic model

    Args:
        config_class: Pydantic BaseModel subclass to parse into
        config_file_path: Path to config file (JSON/YAML/TOML)
        **kwargs: CLI overrides to merge with file config

    Returns:
        Validated instance of config_class

    Raises:
        ValidationError: If configuration is invalid

    Example:
        >>> from pydantic import BaseModel, Field
        >>> class MyConfig(BaseModel):
        ...     name: str = Field(...)
        ...     value: int = Field(default=0)
        >>> cfg = load_and_parse_config(MyConfig, 'config.json', value=42)
        >>> assert isinstance(cfg, MyConfig)

    """
    raw_config = load_cli_config(config_file_path, **kwargs)
    return config_class(**raw_config)
