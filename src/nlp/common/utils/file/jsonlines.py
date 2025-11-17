from pathlib import Path

import jsonlines


def load_jsonlines(path: str | Path) -> list[dict[str, object]]:
    with jsonlines.open(str(Path(path))) as reader:
        return [dict(entry) for entry in reader]


def save_as_jsonlines(
    data: list[dict[str, object]],
    path: str | Path,
    parents: bool = True,
    exist_ok: bool = True,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=parents, exist_ok=exist_ok)
    with jsonlines.open(str(target), mode='w') as writer:
        for datum in data:
            writer.write(datum)
