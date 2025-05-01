import json
from pathlib import Path


def load_json(path: str | Path) -> dict | list:
    with Path(path).open(mode='r', encoding='utf-8') as fin:
        data = json.load(fin)
    return data


def save_as_indented_json(
    data: dict | list,
    path: str | Path,
    parents: bool = True,
    exist_ok: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=parents, exist_ok=exist_ok)
    with path.open(mode='w', encoding='utf-8') as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4, separators=(',', ': '))
    return
