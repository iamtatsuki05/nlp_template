"""Behavioral tests for text cleaning utilities."""

import pytest

from nlp.constract_llm.dataset.cleanse.sample import cleanse_sample
from nlp.constract_llm.dataset.cleanse.text import (
    TextCleaner,
    create_text_cleaner,
    is_blank,
    is_include_email,
    is_include_url,
    is_only_numeric,
    is_out_of_length_range,
)


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('', True),
        ('   ', True),
        ('abc', False),
    ],
)
def test_is_blank(text: str, expected: bool) -> None:
    assert is_blank(text) == expected


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('123', True),
        ('123abc', False),
        ('', False),
    ],
)
def test_is_only_numeric(text: str, expected: bool) -> None:
    assert is_only_numeric(text) == expected


@pytest.mark.parametrize(
    ('text', 'min_len', 'max_len', 'expected'),
    [
        ('hello', 1, 10, False),
        ('hello', 6, 10, True),
        ('', 1, 5, True),
    ],
)
def test_is_out_of_length_range(text: str, min_len: int, max_len: int, expected: bool) -> None:
    assert is_out_of_length_range(text, min_str_len=min_len, max_str_len=max_len) == expected


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('https://example.com', True),
        ('plain text', False),
    ],
)
def test_is_include_url(text: str, expected: bool) -> None:
    assert is_include_url(text) == expected


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('test@example.com', True),
        ('invalid', False),
    ],
)
def test_is_include_email(text: str, expected: bool) -> None:
    assert is_include_email(text) == expected


def test_text_cleaner_filters_by_rules() -> None:
    cleaner = create_text_cleaner(
        do_rm_time_schedule=False,
        do_rm_only_numeric=True,
        do_rm_include_email_text=False,
        do_rm_include_url_text=True,
    )

    assert cleaner.clean('12345') is None
    assert cleaner.clean('Visit https://example.com') is None
    assert cleaner.clean('keep me') == 'keep me'


def test_cleanse_sample_accepts_custom_cleaner() -> None:
    class AllowAllRule:
        def should_remove(self, _: str) -> bool:
            return False

    cleaner = TextCleaner([AllowAllRule()])
    sample = {'body': '12345'}
    result = cleanse_sample(sample, ['body'], do_rm_only_numeric=True, text_cleaner=cleaner)

    assert result['body'] == '12345'
