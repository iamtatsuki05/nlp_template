import json
from pathlib import Path

from nlp.common.utils.cli_utils import load_cli_config


def test_load_cli_config_merges_file_and_kwargs(tmp_path: Path) -> None:
    config_file = tmp_path / 'config.json'
    config_payload = {'file_key': 'file_value', 'override_key': 'file_value'}
    config_file.write_text(json.dumps(config_payload), encoding='utf-8')

    result = load_cli_config(config_file, override_key='cli_value', new_key='cli_value')

    assert result == {
        'file_key': 'file_value',
        'override_key': 'cli_value',
        'new_key': 'cli_value',
    }


def test_load_cli_config_without_file_returns_kwargs() -> None:
    result = load_cli_config(None, cli_key='cli_value')
    assert result == {'cli_key': 'cli_value'}


def test_load_cli_config_accepts_path_types(tmp_path: Path) -> None:
    payload = {'setting': 'value'}
    config_file = tmp_path / 'config.json'
    config_file.write_text(json.dumps(payload), encoding='utf-8')

    assert load_cli_config(str(config_file)) == payload
    assert load_cli_config(config_file) == payload
