from pathlib import Path
from typing import Any

from project.common.utils.file.json import load_json
from project.common.utils.file.toml import load_toml
from project.common.utils.file.yaml import load_yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load configuration from a file (JSON, YAML, or TOML).
    """
    ext = Path(path).suffix.lower()
    if ext == '.json':
        return load_json(path)
    elif ext in ('.yaml', '.yml'):
        return load_yaml(path)
    elif ext == '.toml':
        return load_toml(path)
    else:
        raise ValueError(f'Unsupported config file format: {ext}')
