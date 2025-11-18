from pathlib import Path
from typing import Any

from nlp.common.utils.file.io import load_file


def load_config(path: str | Path) -> dict[str, Any]:
    """Load configuration from a file (JSON, YAML, TOML, XML).

    Args:
        path: Path to the configuration file. Format is detected from extension.

    Returns:
        Configuration data as a dictionary.

    Raises:
        ValueError: If file format is not supported.
        TypeError: If the loaded data is not a dictionary.

    """
    data = load_file(path)

    if not isinstance(data, dict):
        raise TypeError(f'Config file {path!r} did not return a dict, got {type(data).__name__}')

    return data
