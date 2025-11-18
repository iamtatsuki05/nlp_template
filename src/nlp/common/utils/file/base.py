from pathlib import Path
from typing import Any, Protocol, TypeVar

T = TypeVar('T')

JsonLikeValue = dict[str, Any] | list[Any] | str | int | float | bool | None


class FileLoader(Protocol):
    """Protocol for loading data from files.

    Implementations should handle file format-specific deserialization.
    """

    def load(self, path: str | Path) -> Any:  # noqa: ANN401
        """Load data from the specified file path.

        Args:
            path: Path to the file to load

        Returns:
            Deserialized data from the file

        """
        ...


class FileSaver(Protocol):
    """Protocol for saving data to files.

    Implementations should handle file format-specific serialization.
    """

    def save(
        self,
        data: Any,  # noqa: ANN401
        path: str | Path,
        *,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> None:
        """Save data to the specified file path.

        Args:
            data: Data to serialize and save
            path: Path where the file should be saved
            parents: If True, create parent directories as needed
            exist_ok: If True, don't raise error if directory exists

        """
        ...


class FileHandler(Protocol):
    """Combined protocol for both loading and saving files.

    Provides a complete interface for file I/O operations.
    """

    def load(self, path: str | Path) -> Any:  # noqa: ANN401
        """Load data from the specified file path."""
        ...

    def save(
        self,
        data: Any,  # noqa: ANN401
        path: str | Path,
        *,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> None:
        """Save data to the specified file path."""
        ...
