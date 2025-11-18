from pathlib import Path

import pytest

from nlp.common.utils.file.yaml import YamlValue, load_yaml, save_as_indented_yaml


def test_load_yaml_from_file(tmp_path: Path) -> None:
    yaml_content = """\
key: value
nested:
  number: 42
"""
    yaml_file = tmp_path / 'config.yaml'
    yaml_file.write_text(yaml_content, encoding='utf-8')

    expected: YamlValue = {'key': 'value', 'nested': {'number': 42}}
    assert load_yaml(str(yaml_file)) == expected
    assert load_yaml(yaml_file) == expected


def test_save_and_load_yaml(tmp_path: Path) -> None:
    data: YamlValue = {'message': 'hello', 'count': 3}
    yaml_file = tmp_path / 'nested' / 'config.yaml'

    save_as_indented_yaml(data, yaml_file)
    assert yaml_file.exists()

    loaded = load_yaml(yaml_file)
    assert loaded == data

    content = yaml_file.read_text(encoding='utf-8')
    assert 'message:' in content
    assert 'count:' in content


def test_save_yaml_without_parents_raises(tmp_path: Path) -> None:
    data: YamlValue = {'flag': True}
    yaml_file = tmp_path / 'level1' / 'level2' / 'config.yaml'

    with pytest.raises(FileNotFoundError):
        save_as_indented_yaml(data, yaml_file, parents=False)
