import pytest

from nlp.constract_llm.dataset.preprocess.preprocess import clean_text


@pytest.mark.parametrize(
    ('input_text', 'expected_output'),
    [
        ('Hello, World!', 'Hello, World!'),
        ('  Hello   World!  ', 'Hello World!'),
        ('\tHello\nWorld!\t', 'Hello World!'),
        ('Unicode: ñ, ü, é', 'Unicode: ñ, ü, é'),
        ('Special chars: @#$%^&*()', 'Special chars: @#$%^&*()'),
        ('', ''),
        ('   ', ''),
    ],
)
def test_clean_text(input_text: str, expected_output: str) -> None:
    """Test the clean_text function."""
    result = clean_text(input_text)
    assert result == expected_output
