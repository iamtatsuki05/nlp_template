from pathlib import Path

import jsonlines


def load_jsonlines(path: str | Path) -> list[dict]:
    data_list = []
    with jsonlines.open(str(path)) as reader:
        for data in reader:
            data_list.append(data)
    return data_list


def save_as_jsonlines(
    data: list[dict],
    path: str | Path,
    parents: bool = True,
    exist_ok: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=parents, exist_ok=exist_ok)
    with jsonlines.open(str(path), mode='w') as writer:
        for datum in data:
            writer.write(datum)
    return
