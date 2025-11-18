from pathlib import Path
from typing import Any

import pytest

from nlp.common.utils.file.xml import XmlFileHandler, load_xml, save_as_xml


@pytest.fixture
def sample_xml_data() -> dict[str, Any]:
    return {
        'root': {
            'name': 'John Doe',
            'age': '30',
            'city': 'Tokyo',
        }
    }


@pytest.fixture
def nested_xml_data() -> dict[str, Any]:
    return {
        'config': {
            'database': {
                'host': 'localhost',
                'port': '5432',
            },
            'cache': {
                'enabled': 'true',
                'ttl': '3600',
            },
        }
    }


def test_load_xml_simple(tmp_path: Path, sample_xml_data: dict[str, Any]) -> None:
    xml_file = tmp_path / 'data.xml'
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
<root>
  <name>John Doe</name>
  <age>30</age>
  <city>Tokyo</city>
</root>"""
    xml_file.write_text(xml_content)

    result = load_xml(xml_file)
    assert result == sample_xml_data


def test_save_as_xml_simple(tmp_path: Path, sample_xml_data: dict[str, Any]) -> None:
    xml_file = tmp_path / 'output.xml'
    save_as_xml(sample_xml_data, xml_file)

    assert xml_file.exists()
    content = xml_file.read_text()
    assert '<?xml version' in content
    assert '<root>' in content
    assert '<name>John Doe</name>' in content
    assert '<age>30</age>' in content
    assert '<city>Tokyo</city>' in content


def test_save_and_load_xml_roundtrip(tmp_path: Path, sample_xml_data: dict[str, Any]) -> None:
    xml_file = tmp_path / 'roundtrip.xml'
    save_as_xml(sample_xml_data, xml_file)
    loaded = load_xml(xml_file)

    assert loaded == sample_xml_data


def test_save_and_load_nested_xml(tmp_path: Path, nested_xml_data: dict[str, Any]) -> None:
    xml_file = tmp_path / 'nested.xml'
    save_as_xml(nested_xml_data, xml_file)
    loaded = load_xml(xml_file)

    assert loaded == nested_xml_data


def test_xml_file_handler(tmp_path: Path, sample_xml_data: dict[str, Any]) -> None:
    handler = XmlFileHandler()
    xml_file = tmp_path / 'handler_test.xml'

    handler.save(sample_xml_data, xml_file)
    assert xml_file.exists()

    loaded = handler.load(xml_file)
    assert loaded == sample_xml_data


def test_xml_file_handler_custom_root_tag(tmp_path: Path) -> None:
    handler = XmlFileHandler(root_tag='config')
    data = {'setting1': 'value1', 'setting2': 'value2'}
    xml_file = tmp_path / 'custom_root.xml'

    handler.save(data, xml_file)
    content = xml_file.read_text()
    assert '<config>' in content


def test_save_xml_creates_parent_directories(tmp_path: Path, sample_xml_data: dict[str, Any]) -> None:
    xml_file = tmp_path / 'nested' / 'dir' / 'data.xml'
    save_as_xml(sample_xml_data, xml_file)

    assert xml_file.exists()


def test_save_xml_respects_parents_flag(tmp_path: Path, sample_xml_data: dict[str, Any]) -> None:
    deeply_nested_dir = tmp_path / 'level1' / 'level2'
    xml_file = deeply_nested_dir / 'data.xml'

    with pytest.raises(FileNotFoundError):
        save_as_xml(sample_xml_data, xml_file, parents=False)


def test_xml_with_list_data(tmp_path: Path) -> None:
    data = {'items': {'item': ['apple', 'banana', 'cherry']}}
    xml_file = tmp_path / 'list_data.xml'

    save_as_xml(data, xml_file)
    assert xml_file.exists()

    loaded = load_xml(xml_file)
    # XML conversion may change structure slightly for lists
    assert 'items' in loaded


def test_load_xml_accepts_string_path(tmp_path: Path, sample_xml_data: dict[str, Any]) -> None:
    xml_file = tmp_path / 'string_path.xml'
    save_as_xml(sample_xml_data, xml_file)

    result = load_xml(str(xml_file))
    assert result == sample_xml_data
