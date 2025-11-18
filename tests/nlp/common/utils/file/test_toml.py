from pathlib import Path

import pytest
import toml

from nlp.common.utils.file.toml import load_toml, save_as_toml


def test_load_toml_from_file(tmp_path: Path) -> None:
    content = """\
title = "Example"
[nested]
value = 1
"""
    toml_file = tmp_path / 'config.toml'
    toml_file.write_text(content, encoding='utf-8')

    expected = {'title': 'Example', 'nested': {'value': 1}}
    assert load_toml(str(toml_file)) == expected
    assert load_toml(toml_file) == expected


def test_save_and_load_toml(tmp_path: Path) -> None:
    data = {'service': {'host': 'localhost', 'port': 8080}, 'flag': True}
    toml_file = tmp_path / 'nested' / 'config.toml'

    save_as_toml(data, toml_file)
    assert toml_file.exists()

    loaded = load_toml(toml_file)
    assert loaded == data

    parsed = toml.load(toml_file)
    assert parsed == data


def test_save_toml_without_parents_raises(tmp_path: Path) -> None:
    data = {'key': 'value'}
    toml_file = tmp_path / 'level1' / 'level2' / 'config.toml'

    with pytest.raises(FileNotFoundError):
        save_as_toml(data, toml_file, parents=False)
