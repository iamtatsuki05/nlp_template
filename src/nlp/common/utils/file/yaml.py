from pathlib import Path
from typing import Any

import yaml

YamlValue = dict[str, Any] | list[Any] | str | int | float | bool | None


def load_yaml(path: str | Path) -> YamlValue:
    with Path(path).open(mode='r', encoding='utf-8') as fin:
        return yaml.safe_load(fin)


def save_as_indented_yaml(
    data: YamlValue,
    path: str | Path,
    parents: bool = True,
    exist_ok: bool = True,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=parents, exist_ok=exist_ok)
    with target.open(mode='w', encoding='utf-8') as fout:
        yaml.dump(data, fout, allow_unicode=True, indent=4, default_flow_style=False)


class YamlFileHandler:
    """YAML file handler implementing FileHandler protocol."""

    def load(self, path: str | Path) -> YamlValue:
        """Load YAML data from file."""
        return load_yaml(path)

    def save(
        self,
        data: YamlValue,
        path: str | Path,
        *,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> None:
        """Save data as indented YAML to file."""
        save_as_indented_yaml(data, path, parents=parents, exist_ok=exist_ok)
