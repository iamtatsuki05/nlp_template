from pathlib import Path
from typing import Any

import toml


def load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(mode='r', encoding='utf-8') as fin:
        return toml.load(fin)


def save_as_toml(
    data: dict[str, Any],
    path: str | Path,
    parents: bool = True,
    exist_ok: bool = True,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=parents, exist_ok=exist_ok)
    with target.open(mode='w', encoding='utf-8') as fout:
        toml.dump(data, fout)


class TomlFileHandler:
    """TOML file handler implementing FileHandler protocol."""

    def load(self, path: str | Path) -> dict[str, Any]:
        """Load TOML data from file."""
        return load_toml(path)

    def save(
        self,
        data: dict[str, Any],
        path: str | Path,
        *,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> None:
        """Save data as TOML to file."""
        save_as_toml(data, path, parents=parents, exist_ok=exist_ok)
