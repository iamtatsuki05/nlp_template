import importlib
from pathlib import Path

import pytest

from nlp.common.utils.import_utils import get_imported_function_path, import_function


def test_import_function_returns_target_loader() -> None:
    target_path = Path('src/nlp/common/utils/file/json.py')
    function = import_function(str(target_path), 'load_json')

    module = importlib.import_module('src.nlp.common.utils.file.json')
    assert function is module.load_json


def test_import_function_default_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_dir = tmp_path / 'project'
    project_dir.mkdir()
    module_file = project_dir / 'dynamic_module.py'
    module_file.write_text(
        "def dynamic_module():\n    return 'dynamic result'\n",
        encoding='utf-8',
    )

    monkeypatch.chdir(project_dir)

    function = import_function(str(module_file))
    assert function() == 'dynamic result'


def test_get_imported_function_path() -> None:
    def dummy_function() -> None:
        pass

    file_path = get_imported_function_path(dummy_function)
    assert Path(file_path).resolve() == Path(__file__).resolve()
