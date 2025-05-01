from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict | list:
    with Path(path).open(mode='r', encoding='utf-8') as fin:
        data = yaml.safe_load(fin)
    return data


def save_as_indented_yaml(
    data: dict | list,
    path: str | Path,
    parents: bool = True,
    exist_ok: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=parents, exist_ok=exist_ok)
    with path.open(mode='w', encoding='utf-8') as fout:
        yaml.dump(data, fout, allow_unicode=True, indent=4, default_flow_style=False)
    return
