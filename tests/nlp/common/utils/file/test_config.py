from pathlib import Path

import pytest

from nlp.common.utils.file.config import load_config


def test_load_config_json(tmp_path: Path) -> None:
    config_file = tmp_path / 'config.json'
    config_file.write_text('{"key": "value", "number": 42}')

    result = load_config(config_file)
    assert result == {'key': 'value', 'number': 42}


def test_load_config_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / 'config.yaml'
    config_file.write_text('key: value\nnumber: 42')

    result = load_config(config_file)
    assert result == {'key': 'value', 'number': 42}


def test_load_config_toml(tmp_path: Path) -> None:
    config_file = tmp_path / 'config.toml'
    config_file.write_text('key = "value"\nnumber = 42')

    result = load_config(config_file)
    assert result == {'key': 'value', 'number': 42}


def test_load_config_xml(tmp_path: Path) -> None:
    config_file = tmp_path / 'config.xml'
    config_file.write_text("""<?xml version="1.0" encoding="utf-8"?>
<config>
  <key>value</key>
  <number>42</number>
</config>""")

    result = load_config(config_file)
    assert 'config' in result
    assert result['config']['key'] == 'value'
    assert result['config']['number'] == '42'


def test_load_config_unsupported_format(tmp_path: Path) -> None:
    config_file = tmp_path / 'config.txt'
    config_file.write_text('some text content')

    with pytest.raises(ValueError, match='Unsupported file extension'):
        load_config(config_file)


def test_load_config_not_dict(tmp_path: Path) -> None:
    config_file = tmp_path / 'config.json'
    config_file.write_text('["not", "a", "dict"]')

    with pytest.raises(TypeError, match='did not return a dict'):
        load_config(config_file)


def test_load_config_accepts_string_path(tmp_path: Path) -> None:
    config_file = tmp_path / 'config.json'
    config_file.write_text('{"key": "value"}')

    result = load_config(str(config_file))
    assert result == {'key': 'value'}
