"""Generic file I/O operations using FileHandler abstraction.

This module provides format-agnostic file operations that automatically
detect and handle different file formats (JSON, YAML, TOML).
"""

from pathlib import Path
from typing import Any

from nlp.common.utils.file.factory import get_file_handler


def load_file(path: str | Path) -> Any:  # noqa: ANN401
    """Load data from a file, automatically detecting format from extension.

    Args:
        path: Path to the file (extension determines format)

    Returns:
        Deserialized data from the file

    Raises:
        ValueError: If file format cannot be detected or is unsupported

    Example:
        >>> data = load_file('config.json')
        >>> data = load_file('settings.yaml')
        >>> data = load_file('pyproject.toml')

    """
    handler = get_file_handler(path)
    return handler.load(path)


def save_file(
    data: Any,  # noqa: ANN401
    path: str | Path,
    *,
    parents: bool = True,
    exist_ok: bool = True,
) -> None:
    """Save data to a file, automatically detecting format from extension.

    Args:
        data: Data to save
        path: Path where the file should be saved (extension determines format)
        parents: If True, create parent directories as needed
        exist_ok: If True, don't raise error if directory exists

    Raises:
        ValueError: If file format cannot be detected or is unsupported

    Example:
        >>> save_file({'key': 'value'}, 'output.json')
        >>> save_file(['item1', 'item2'], 'output.yaml')
        >>> save_file({'tool': {'poetry': {}}}, 'pyproject.toml')

    """
    handler = get_file_handler(path)
    handler.save(data, path, parents=parents, exist_ok=exist_ok)
