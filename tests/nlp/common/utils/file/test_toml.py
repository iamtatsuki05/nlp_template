from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from nlp.common.utils.file.toml import load_toml, save_as_toml


@pytest.mark.parametrize(
    'input_data,expected_result',
    [
        ('key = "value"', {'key': 'value'}),
        ('[nested]\nkey = "value"', {'nested': {'key': 'value'}}),
        (
            '[array]\nvalues = ["item1", "item2"]',
            {'array': {'values': ['item1', 'item2']}},
        ),
        ('', {}),
    ],
)
def test_load_toml(input_data, expected_result):
    """Test that load_toml correctly loads and parses TOML data."""
    # Mock the open function to return our test data
    with patch('pathlib.Path.open', mock_open(read_data=input_data)):
        # Test with string path
        result_str = load_toml('dummy/path.toml')
        assert result_str == expected_result

        # Test with Path object
        result_path = load_toml(Path('dummy/path.toml'))
        assert result_path == expected_result


@pytest.mark.parametrize(
    'input_data',
    [
        ({'key': 'value'}),
        ({'nested': {'key': 'value'}}),
        ({'array': {'values': ['item1', 'item2']}}),
        ({}),
    ],
)
def test_save_as_toml(input_data):
    """Test that save_as_toml correctly writes TOML data to a file."""
    mock_file = mock_open()

    # Create a patch for both the open function, mkdir, and toml.dump
    with (
        patch('pathlib.Path.open', mock_file),
        patch('pathlib.Path.mkdir') as mock_mkdir,
        patch('toml.dump') as mock_dump,
    ):
        # Test with string path
        save_as_toml(input_data, 'dummy/path.toml')

        # Verify mkdir was called with the expected parameters
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify that the file was opened in write mode
        mock_file.assert_called_once_with(mode='w', encoding='utf-8')

        # Get the handle to the mock file
        handle = mock_file()

        # Verify toml.dump was called with the correct arguments
        mock_dump.assert_called_once_with(input_data, handle)


def test_save_as_toml_path_object():
    """Test save_as_toml with a Path object."""
    mock_file = mock_open()
    test_data = {'key': 'value'}

    with (
        patch('pathlib.Path.open', mock_file),
        patch('pathlib.Path.mkdir') as mock_mkdir,
    ):
        # Test with Path object
        save_as_toml(test_data, Path('dummy/path.toml'))

        # Verify mkdir and open were called
        mock_mkdir.assert_called_once()
        mock_file.assert_called_once()

        # Verify content was written
        handle = mock_file()
        assert handle.write.called
