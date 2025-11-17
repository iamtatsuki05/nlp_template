import os
import sys
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from nlp.common.utils.import_utils import get_imported_function_path, import_function


@pytest.fixture
def test_module_file(tmp_path: Path) -> Generator[str]:
    """Create a temporary Python module for testing import functions."""
    module_dir = tmp_path / 'test_module'
    module_dir.mkdir()
    (module_dir / '__init__.py').write_text('')

    module_file = module_dir / 'test_func.py'
    module_file.write_text(
        'def test_function():\n'
        "    return 'Hello from test_function'\n"
        '\n'
        'def another_function():\n'
        "    return 'Hello from another_function'\n"
    )

    # Add the tmp_path to sys.path temporarily
    sys.path.insert(0, str(tmp_path.parent))
    yield str(module_file)

    # Clean up
    sys.path.remove(str(tmp_path.parent))


@pytest.mark.parametrize(
    ('function_name', 'expected_result'),
    [
        ('test_function', 'Hello from test_function'),
        ('another_function', 'Hello from another_function'),
    ],
)
def test_import_function(test_module_file: str, function_name: str, expected_result: str) -> None:
    # Handle the default case where function_name is None
    if function_name is None:
        # Mock the stem attribute to return "test_func"
        with patch('pathlib.Path') as mock_path:
            mock_path_instance = mock_path.return_value
            mock_path_instance.resolve.return_value = Path(test_module_file)
            mock_path_instance.stem = 'test_function'

            # Mock the current working directory
            with patch('pathlib.Path.cwd') as mock_cwd:
                mock_cwd.return_value = Path(test_module_file).parent.parent

                # Add mock for relative_to
                mock_path_instance.relative_to.return_value = Path('test_module/test_func.py')
                mock_path_instance.with_suffix.return_value = Path('test_module/test_func')

                function = import_function(test_module_file, function_name)
    else:
        # For normal cases where function_name is provided
        with patch('pathlib.Path.cwd') as mock_cwd:
            mock_cwd.return_value = Path(test_module_file).parent.parent
            function = import_function(test_module_file, function_name)

    # Assert that the imported function returns the expected result
    assert function() == expected_result


def test_get_imported_function_path() -> None:
    # Create a dummy function for testing
    def dummy_function() -> None:
        pass

    # Get the file path of the dummy function
    file_path = get_imported_function_path(dummy_function)

    # Assert that the returned path is the path of this test file
    assert os.path.realpath(file_path) == os.path.realpath(__file__)
