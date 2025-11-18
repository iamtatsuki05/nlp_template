from pathlib import Path
from typing import Any

from nlp.common.utils.file.json import load_json
from nlp.common.utils.file.toml import load_toml
from nlp.common.utils.file.yaml import load_yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load configuration from a file (JSON, YAML, or TOML)."""
    ext = Path(path).suffix.lower()

    if ext == '.json':
        data = load_json(path)
    elif ext in ('.yaml', '.yml'):
        data = load_yaml(path)
    elif ext == '.toml':
        data = load_toml(path)
    else:
        raise ValueError(f'Unsupported config file format: {ext}')

    if not isinstance(data, dict):
        raise TypeError(f'Config file {path!r} did not return a dict, got {type(data).__name__}')

    return data
