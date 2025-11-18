from pathlib import Path
from typing import Any

import pytest

from nlp.common.utils.file.io import load_file, save_file


@pytest.fixture
def sample_data() -> dict[str, Any]:
    return {'key': 'value', 'number': 42, 'list': [1, 2, 3]}


def test_save_and_load_json(tmp_path: Path, sample_data: dict[str, Any]) -> None:
    file_path = tmp_path / 'test.json'
    save_file(sample_data, file_path)

    assert file_path.exists()
    loaded = load_file(file_path)
    assert loaded == sample_data


def test_save_and_load_yaml(tmp_path: Path, sample_data: dict[str, Any]) -> None:
    file_path = tmp_path / 'test.yaml'
    save_file(sample_data, file_path)

    assert file_path.exists()
    loaded = load_file(file_path)
    assert loaded == sample_data


def test_save_and_load_yml(tmp_path: Path, sample_data: dict[str, Any]) -> None:
    file_path = tmp_path / 'test.yml'
    save_file(sample_data, file_path)

    assert file_path.exists()
    loaded = load_file(file_path)
    assert loaded == sample_data


def test_save_and_load_toml(tmp_path: Path, sample_data: dict[str, Any]) -> None:
    file_path = tmp_path / 'test.toml'
    save_file(sample_data, file_path)

    assert file_path.exists()
    loaded = load_file(file_path)
    assert loaded == sample_data


def test_save_creates_parent_directories(tmp_path: Path, sample_data: dict[str, Any]) -> None:
    file_path = tmp_path / 'nested' / 'dir' / 'test.json'
    save_file(sample_data, file_path)

    assert file_path.exists()
    loaded = load_file(file_path)
    assert loaded == sample_data


def test_save_respects_parents_flag(tmp_path: Path, sample_data: dict[str, Any]) -> None:
    """Test that parents=False prevents creation of nested directories."""
    deeply_nested_dir = tmp_path / 'level1' / 'level2'
    file_path = deeply_nested_dir / 'test.json'

    # Multiple levels of directories don't exist, expect FileNotFoundError
    with pytest.raises(FileNotFoundError):
        save_file(sample_data, file_path, parents=False)

    # With parents=True, nested directories are created
    save_file(sample_data, file_path, parents=True)
    assert file_path.exists()


def test_load_unsupported_format(tmp_path: Path) -> None:
    file_path = tmp_path / 'test.xml'
    file_path.write_text('<root></root>')

    with pytest.raises(ValueError, match='Unsupported file extension'):
        load_file(file_path)


def test_save_unsupported_format(tmp_path: Path, sample_data: dict[str, Any]) -> None:
    file_path = tmp_path / 'test.xml'

    with pytest.raises(ValueError, match='Unsupported file extension'):
        save_file(sample_data, file_path)


def test_json_indentation(tmp_path: Path, sample_data: dict[str, Any]) -> None:
    file_path = tmp_path / 'test.json'
    save_file(sample_data, file_path)

    content = file_path.read_text()
    assert '    ' in content  # 4-space indentation


def test_yaml_formatting(tmp_path: Path, sample_data: dict[str, Any]) -> None:
    file_path = tmp_path / 'test.yaml'
    save_file(sample_data, file_path)

    content = file_path.read_text()
    assert 'key: value' in content


def test_roundtrip_preserves_data_types(tmp_path: Path) -> None:
    """Test that data types are preserved across save/load cycles."""
    common_data = {
        'string': 'text',
        'int': 42,
        'float': 3.14,
        'bool': True,
        'list': [1, 2, 3],
        'nested': {'inner': 'value'},
    }

    # Test JSON and YAML with null values (TOML doesn't support null)
    data_with_null = {**common_data, 'null': None}
    for ext in ['json', 'yaml']:
        file_path = tmp_path / f'test.{ext}'
        save_file(data_with_null, file_path)
        loaded = load_file(file_path)
        assert loaded == data_with_null

    # Test TOML without null values
    toml_path = tmp_path / 'test.toml'
    save_file(common_data, toml_path)
    loaded = load_file(toml_path)
    assert loaded == common_data
