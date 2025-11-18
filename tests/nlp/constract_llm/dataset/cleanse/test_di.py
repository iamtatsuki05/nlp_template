"""Simple behavior-driven tests for text cleaner DI."""

import pytest
from injector import Injector

from nlp.constract_llm.dataset.cleanse.cleaner import TextCleaner
from nlp.constract_llm.dataset.cleanse.di import TextCleanerModule, create_text_cleaner_via_di


def test_text_cleaner_module_respects_rules() -> None:
    module = TextCleanerModule(
        do_rm_time_schedule=False,
        do_rm_only_numeric=True,
        do_rm_include_url_text=True,
        do_rm_include_email_text=False,
    )
    cleaner = Injector([module]).get(TextCleaner)

    assert cleaner.clean('123') is None
    assert cleaner.clean('  Visit https://example.com  ') is None
    assert cleaner.clean('regular text') == 'regular text'


@pytest.mark.parametrize(
    ('settings', 'text', 'expected'),
    [
        ({'do_rm_only_numeric': True}, '123', None),
        ({'do_rm_only_numeric': False}, '123', '123'),
        ({'do_rm_include_url_text': True}, 'https://example.com', None),
        ({'do_rm_include_url_text': False}, 'https://example.com', 'https://example.com'),
    ],
)
def test_create_text_cleaner_via_di_applies_expected_rules(
    settings: dict[str, bool],
    text: str,
    expected: str | None,
) -> None:
    cleaner = create_text_cleaner_via_di(**settings)
    assert cleaner.clean(text) == expected
