from pathlib import Path
from unittest.mock import patch

import pytest

from nlp.common.utils.cli_utils import load_cli_config


@pytest.mark.parametrize(
    'config_file_exists,config_content,kwargs,expected',
    [
        # Case 1: Config file exists with content, no kwargs
        (True, {'file_key': 'file_value'}, {}, {'file_key': 'file_value'}),
        # Case 2: Config file exists with content, with kwargs
        (
            True,
            {'file_key': 'file_value', 'override_key': 'file_value'},
            {'override_key': 'cli_value', 'new_key': 'cli_value'},
            {
                'file_key': 'file_value',
                'override_key': 'cli_value',
                'new_key': 'cli_value',
            },
        ),
        # Case 3: No config file, with kwargs
        (False, None, {'cli_key': 'cli_value'}, {'cli_key': 'cli_value'}),
        # Case 4: No config file, no kwargs
        (False, None, {}, {}),
    ],
)
def test_load_cli_config(config_file_exists, config_content, kwargs, expected):
    mock_path = 'path/to/config.json' if config_file_exists else None

    # Mock the load_config function directly within the cli_utils module
    with patch('nlp.common.utils.cli_utils.load_config') as mock_load_config:
        mock_load_config.return_value = config_content

        # Call the function under test
        result = load_cli_config(mock_path, **kwargs)

        # Verify the results
        assert result == expected

        # Verify that load_config was called appropriately
        if config_file_exists:
            mock_load_config.assert_called_once_with(mock_path)
        else:
            mock_load_config.assert_not_called()


@pytest.mark.parametrize(
    'config_file_path,expected_path_type',
    [
        # Test with string path
        ('path/to/config.json', str),
        # Test with Path object
        (Path('path/to/config.json'), Path),
    ],
)
def test_load_cli_config_path_types(config_file_path, expected_path_type):
    # Create a comprehensive mock to prevent file system access by mocking where the function is used
    with patch('nlp.common.utils.cli_utils.load_config') as mock_load_config:
        mock_load_config.return_value = {}

        # Ensure we don't actually try to open a file
        load_cli_config(config_file_path)

        # Verify that load_config was called with the correct path type
        mock_load_config.assert_called_once()
        args, _ = mock_load_config.call_args
        assert isinstance(args[0], expected_path_type)
