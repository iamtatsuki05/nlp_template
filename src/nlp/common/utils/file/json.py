import json
from pathlib import Path
from typing import Any

JsonValue = dict[Any, Any] | list[Any] | str | int | float | bool | None


def load_json(path: str | Path) -> JsonValue:
    with Path(path).open(mode='r', encoding='utf-8') as fin:
        return json.load(fin)


def save_as_indented_json(
    data: JsonValue,
    path: str | Path,
    parents: bool = True,
    exist_ok: bool = True,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=parents, exist_ok=exist_ok)
    with target.open(mode='w', encoding='utf-8') as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4, separators=(',', ': '))


class JsonFileHandler:
    """JSON file handler implementing FileHandler protocol."""

    def load(self, path: str | Path) -> JsonValue:
        """Load JSON data from file."""
        return load_json(path)

    def save(
        self,
        data: JsonValue,
        path: str | Path,
        *,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> None:
        """Save data as indented JSON to file."""
        save_as_indented_json(data, path, parents=parents, exist_ok=exist_ok)
