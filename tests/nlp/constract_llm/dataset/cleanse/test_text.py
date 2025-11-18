import pytest

from nlp.constract_llm.dataset.cleanse.text import (
    is_blank,
    is_include_email,
    is_include_url,
    is_only_numeric,
    is_out_of_length_range,
)


@pytest.mark.parametrize(
    ('text', 'expected_result'),
    [
        ('Hello, World!', False),
        ('   ', True),
        ('\t\n', True),
        ('', True),
    ],
)
def test_is_blank(text: str, expected_result: bool) -> None:
    """Test the is_blank function."""
    result = is_blank(text)
    assert result == expected_result


@pytest.mark.parametrize(
    ('text', 'expected_result'),
    [
        ('123456', True),
        ('abc123', False),
        ('123abc', False),
        ('', False),
    ],
)
def test_is_only_numeric(text: str, expected_result: bool) -> None:
    """Test the is_only_numeric function."""
    result = is_only_numeric(text)
    assert result == expected_result


@pytest.mark.parametrize(
    ('text', 'min_length', 'max_length', 'expected_result'),
    [
        ('Hello', 1, 10, False),
        ('Hello', 6, 10, True),
        ('Hello', 1, 4, True),
        ('', 1, 10, True),
    ],
)
def test_is_out_of_length_range(text: str, min_length: int, max_length: int, expected_result: bool) -> None:
    """Test the is_out_of_length_range function."""
    result = is_out_of_length_range(text, min_length, max_length)
    assert result == expected_result


@pytest.mark.parametrize(
    ('text', 'expected_result'),
    [
        ('http://example.com', True),
        ('https://example.com', True),
        ('not_a_url', False),
        ('', False),
    ],
)
def test_is_include_url(text: str, expected_result: bool) -> None:
    """Test the is_include_url function."""
    result = is_include_url(text)
    assert result == expected_result


@pytest.mark.parametrize(
    ('text', 'expected_result'),
    [
        ('hoge@example.com', True),
        ('test@domain.com', True),
        ('invalid-email', False),
        ('', False),
    ],
)
def test_is_include_email(text: str, expected_result: bool) -> None:
    """Test the is_include_email function."""
    result = is_include_email(text)
    assert result == expected_result
