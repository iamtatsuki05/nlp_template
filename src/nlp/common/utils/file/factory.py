from pathlib import Path
from typing import ClassVar, Literal

from nlp.common.utils.file.base import FileHandler
from nlp.common.utils.file.json import JsonFileHandler
from nlp.common.utils.file.toml import TomlFileHandler
from nlp.common.utils.file.yaml import YamlFileHandler

FileFormat = Literal['json', 'yaml', 'toml']


class FileHandlerFactory:
    """Factory for creating file handlers based on file format."""

    _handlers: ClassVar[dict[FileFormat, type[FileHandler]]] = {
        'json': JsonFileHandler,
        'yaml': YamlFileHandler,
        'toml': TomlFileHandler,
    }

    @classmethod
    def create(cls, format_type: FileFormat) -> FileHandler:
        """Create a file handler for the specified format.

        Args:
            format_type: File format ('json', 'yaml', or 'toml')

        Returns:
            File handler instance for the specified format

        Raises:
            ValueError: If format_type is not supported

        """
        handler_class = cls._handlers.get(format_type)
        if handler_class is None:
            supported = ', '.join(cls._handlers.keys())
            msg = f'Unsupported file format: {format_type}. Supported formats: {supported}'
            raise ValueError(msg)
        return handler_class()

    @classmethod
    def from_path(cls, path: str | Path) -> FileHandler:
        """Create a file handler by detecting format from file extension.

        Args:
            path: File path with extension

        Returns:
            File handler instance for the detected format

        Raises:
            ValueError: If file extension is not recognized or missing

        """
        suffix = Path(path).suffix.lstrip('.')
        if not suffix:
            msg = f'Cannot detect file format: no extension in {path}'
            raise ValueError(msg)

        # Map common extensions to format types
        extension_map: dict[str, FileFormat] = {
            'json': 'json',
            'yaml': 'yaml',
            'yml': 'yaml',
            'toml': 'toml',
        }

        format_type = extension_map.get(suffix.lower())
        if format_type is None:
            supported = ', '.join(extension_map.keys())
            msg = f'Unsupported file extension: .{suffix}. Supported extensions: {supported}'
            raise ValueError(msg)

        return cls.create(format_type)


def get_file_handler(path: str | Path) -> FileHandler:
    """Get a file handler from a file path.

    Args:
        path: File path with extension

    Returns:
        File handler instance for the detected format

    """
    return FileHandlerFactory.from_path(path)
