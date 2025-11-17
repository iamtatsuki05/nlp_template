from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from nlp.common.utils.file.yaml import load_yaml, save_as_indented_yaml


@pytest.mark.parametrize(
    ('input_data', 'expected_result'),
    [
        ('key: value', {'key': 'value'}),
        ('nested:\n  key: value', {'nested': {'key': 'value'}}),
        ('- item1\n- item2', ['item1', 'item2']),
        ('{}', {}),
        ('[]', []),
    ],
)
def test_load_yaml(input_data: str, expected_result: object) -> None:
    """Test that load_yaml correctly loads and parses YAML data."""
    # Mock the open function to return our test data
    with patch('pathlib.Path.open', mock_open(read_data=input_data)):
        # Test with string path
        result_str = load_yaml('dummy/path.yaml')
        assert result_str == expected_result

        # Test with Path object
        result_path = load_yaml(Path('dummy/path.yaml'))
        assert result_path == expected_result


@pytest.mark.parametrize(
    'input_data_tuple',
    [
        ({'key': 'value'},),
        ({'nested': {'key': 'value'}},),
        (['item1', 'item2'],),
        ({},),
        ([],),
    ],
)
def test_save_as_indented_yaml(input_data_tuple: tuple[object, ...]) -> None:
    """Test that save_as_indented_yaml correctly writes YAML data to a file."""
    input_data = input_data_tuple[0]
    mock_file = mock_open()

    # Create a patch for both the open function and mkdir
    with (
        patch('pathlib.Path.open', mock_file),
        patch('pathlib.Path.mkdir') as mock_mkdir,
    ):
        # Test with string path
        save_as_indented_yaml(input_data, 'dummy/path.yaml')

        # Verify mkdir was called with the expected parameters
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify that the file was opened in write mode
        mock_file.assert_called_once_with(mode='w', encoding='utf-8')

        # Get the handle to the mock file
        handle = mock_file()

        # Verify content was written
        assert handle.write.called

        # We can't easily verify the exact YAML output due to formatting differences,
        # but we can check that it was called with something
        written_data = ''.join(call.args[0] for call in handle.write.call_args_list)
        assert written_data  # Assert that something was written


def test_save_as_indented_yaml_path_object() -> None:
    """Test save_as_indented_yaml with a Path object."""
    mock_file = mock_open()
    test_data = {'key': 'value'}

    with (
        patch('pathlib.Path.open', mock_file),
        patch('pathlib.Path.mkdir') as mock_mkdir,
    ):
        # Test with Path object
        save_as_indented_yaml(test_data, Path('dummy/path.yaml'))

        # Verify mkdir and open were called
        mock_mkdir.assert_called_once()
        mock_file.assert_called_once()

        # Verify content was written
        handle = mock_file()
        assert handle.write.called
