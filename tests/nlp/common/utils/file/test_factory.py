import pytest

from nlp.common.utils.file.factory import FileHandlerFactory, get_file_handler
from nlp.common.utils.file.json import JsonFileHandler
from nlp.common.utils.file.toml import TomlFileHandler
from nlp.common.utils.file.xml import XmlFileHandler
from nlp.common.utils.file.yaml import YamlFileHandler


def test_create_json_handler() -> None:
    handler = FileHandlerFactory.create('json')
    assert isinstance(handler, JsonFileHandler)


def test_create_yaml_handler() -> None:
    handler = FileHandlerFactory.create('yaml')
    assert isinstance(handler, YamlFileHandler)


def test_create_toml_handler() -> None:
    handler = FileHandlerFactory.create('toml')
    assert isinstance(handler, TomlFileHandler)


def test_create_xml_handler() -> None:
    handler = FileHandlerFactory.create('xml')
    assert isinstance(handler, XmlFileHandler)


def test_create_unsupported_format() -> None:
    with pytest.raises(ValueError, match='Unsupported file format'):
        FileHandlerFactory.create('txt')  # type: ignore[arg-type]


def test_from_path_json() -> None:
    handler = FileHandlerFactory.from_path('config.json')
    assert isinstance(handler, JsonFileHandler)


def test_from_path_yaml() -> None:
    handler = FileHandlerFactory.from_path('config.yaml')
    assert isinstance(handler, YamlFileHandler)


def test_from_path_yml() -> None:
    handler = FileHandlerFactory.from_path('config.yml')
    assert isinstance(handler, YamlFileHandler)


def test_from_path_toml() -> None:
    handler = FileHandlerFactory.from_path('pyproject.toml')
    assert isinstance(handler, TomlFileHandler)


def test_from_path_xml() -> None:
    handler = FileHandlerFactory.from_path('data.xml')
    assert isinstance(handler, XmlFileHandler)


def test_from_path_no_extension() -> None:
    with pytest.raises(ValueError, match='no extension'):
        FileHandlerFactory.from_path('config')


def test_from_path_unsupported_extension() -> None:
    with pytest.raises(ValueError, match='Unsupported file extension'):
        FileHandlerFactory.from_path('data.txt')


def test_get_file_handler() -> None:
    handler = get_file_handler('settings.json')
    assert isinstance(handler, JsonFileHandler)
