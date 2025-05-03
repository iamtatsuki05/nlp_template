from pathlib import Path

import toml


def load_toml(path: str | Path) -> dict:
    with Path(path).open(mode='r', encoding='utf-8') as fin:
        data = toml.load(fin)
    return data


def save_as_toml(
    data: dict,
    path: str | Path,
    parents: bool = True,
    exist_ok: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=parents, exist_ok=exist_ok)
    with path.open(mode='w', encoding='utf-8') as fout:
        toml.dump(data, fout)
    return
