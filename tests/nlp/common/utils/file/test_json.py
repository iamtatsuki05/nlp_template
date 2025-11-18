import json
from pathlib import Path

import pytest

from nlp.common.utils.file.json import JsonValue, load_json, save_as_indented_json


def test_load_json_from_file(tmp_path: Path) -> None:
    payload = {'name': 'tester', 'values': [1, 2, 3]}
    json_file = tmp_path / 'config.json'
    json_file.write_text(json.dumps(payload), encoding='utf-8')

    assert load_json(str(json_file)) == payload
    assert load_json(json_file) == payload


def test_save_and_load_json(tmp_path: Path) -> None:
    data: JsonValue = {'flag': True, 'count': 5}
    json_file = tmp_path / 'nested' / 'config.json'

    save_as_indented_json(data, json_file)
    assert json_file.exists()

    loaded = load_json(json_file)
    assert loaded == data

    content = json_file.read_text(encoding='utf-8')
    assert content.startswith('{\n')


def test_save_json_without_parents_raises(tmp_path: Path) -> None:
    data: JsonValue = {'value': 'test'}
    json_file = tmp_path / 'level1' / 'level2' / 'config.json'

    with pytest.raises(FileNotFoundError):
        save_as_indented_json(data, json_file, parents=False)
