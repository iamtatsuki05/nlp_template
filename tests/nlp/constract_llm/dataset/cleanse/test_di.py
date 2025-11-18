"""Tests for text cleaner dependency injection module."""

from __future__ import annotations

import pytest
from injector import Injector

from nlp.constract_llm.dataset.cleanse.di import (
    TextCleanerModule,
    create_text_cleaner_via_di,
)
from nlp.constract_llm.dataset.cleanse.text import TextCleaner


class TestTextCleanerModule:
    """Test TextCleanerModule."""

    def test_module_creates_text_cleaner_with_all_rules(self) -> None:
        """Test that module creates TextCleaner with all rules enabled."""
        module = TextCleanerModule(
            do_rm_time_schedule=True,
            rm_time_schedule_threshold=3,
            do_rm_only_numeric=True,
            do_rm_include_url_text=True,
            do_rm_include_email_text=True,
        )
        injector = Injector([module])
        cleaner = injector.get(TextCleaner)

        assert isinstance(cleaner, TextCleaner)
        # Should remove numeric text
        assert cleaner.clean('123.45') is None

    def test_module_creates_text_cleaner_with_no_rules(self) -> None:
        """Test that module creates TextCleaner with no rules enabled."""
        module = TextCleanerModule(
            do_rm_time_schedule=False,
            do_rm_only_numeric=False,
            do_rm_include_url_text=False,
            do_rm_include_email_text=False,
        )
        injector = Injector([module])
        cleaner = injector.get(TextCleaner)

        # Should not remove numeric text when rule is disabled
        result = cleaner.clean('123.45')
        assert result == '123.45'

    def test_module_creates_text_cleaner_with_only_numeric_rule(self) -> None:
        """Test that module creates TextCleaner with only numeric rule."""
        module = TextCleanerModule(
            do_rm_time_schedule=False,
            do_rm_only_numeric=True,
            do_rm_include_url_text=False,
            do_rm_include_email_text=False,
        )
        injector = Injector([module])
        cleaner = injector.get(TextCleaner)

        # Should remove numeric text
        assert cleaner.clean('123') is None
        # Should keep text with URL (URL rule disabled)
        result = cleaner.clean('Check https://example.com')
        assert result == 'Check https://example.com'

    def test_module_creates_text_cleaner_with_url_rule(self) -> None:
        """Test that module creates TextCleaner with URL rule."""
        module = TextCleanerModule(
            do_rm_time_schedule=False,
            do_rm_only_numeric=False,
            do_rm_include_url_text=True,
            do_rm_include_email_text=False,
        )
        injector = Injector([module])
        cleaner = injector.get(TextCleaner)

        # Should remove text with URL
        assert cleaner.clean('Visit https://example.com') is None
        # Should keep regular text
        result = cleaner.clean('Regular text')
        assert result == 'Regular text'

    def test_module_creates_text_cleaner_with_email_rule(self) -> None:
        """Test that module creates TextCleaner with email rule."""
        module = TextCleanerModule(
            do_rm_time_schedule=False,
            do_rm_only_numeric=False,
            do_rm_include_url_text=False,
            do_rm_include_email_text=True,
        )
        injector = Injector([module])
        cleaner = injector.get(TextCleaner)

        # Should remove text with email
        assert cleaner.clean('Contact test@example.com') is None
        # Should keep regular text
        result = cleaner.clean('Regular text')
        assert result == 'Regular text'

    def test_module_singleton_returns_same_instance(self) -> None:
        """Test that module returns the same TextCleaner instance (singleton)."""
        module = TextCleanerModule()
        injector = Injector([module])

        cleaner1 = injector.get(TextCleaner)
        cleaner2 = injector.get(TextCleaner)

        assert cleaner1 is cleaner2


class TestCreateTextCleanerViaDi:
    """Test create_text_cleaner_via_di function."""

    def test_creates_cleaner_with_default_settings(self) -> None:
        """Test that function creates cleaner with default settings."""
        cleaner = create_text_cleaner_via_di()

        assert isinstance(cleaner, TextCleaner)
        # Default should remove numeric text
        assert cleaner.clean('123') is None

    def test_creates_cleaner_with_custom_settings(self) -> None:
        """Test that function creates cleaner with custom settings."""
        cleaner = create_text_cleaner_via_di(
            do_rm_only_numeric=False,
            do_rm_include_url_text=True,
        )

        # Should keep numeric text
        result = cleaner.clean('123')
        assert result == '123'
        # Should remove text with URL
        assert cleaner.clean('Visit https://example.com') is None

    def test_creates_cleaner_with_all_rules_disabled(self) -> None:
        """Test that function creates cleaner with all rules disabled."""
        cleaner = create_text_cleaner_via_di(
            do_rm_time_schedule=False,
            do_rm_only_numeric=False,
            do_rm_include_url_text=False,
            do_rm_include_email_text=False,
        )

        # Should keep all types of text
        assert cleaner.clean('123') == '123'
        assert cleaner.clean('Visit https://example.com') == 'Visit https://example.com'
        assert cleaner.clean('Contact test@example.com') == 'Contact test@example.com'

    def test_creates_cleaner_with_custom_threshold(self) -> None:
        """Test that function creates cleaner with custom time schedule threshold."""
        cleaner = create_text_cleaner_via_di(
            do_rm_time_schedule=True,
            rm_time_schedule_threshold=5,
        )

        assert isinstance(cleaner, TextCleaner)

    @pytest.mark.parametrize(
        ('text', 'expected'),
        [
            ('  Regular text  ', 'Regular text'),  # Strips whitespace
            ('', None),  # Empty text
            ('   ', None),  # Whitespace only
            (None, None),  # None input
        ],
    )
    def test_cleaner_handles_edge_cases(self, text: str | None, expected: str | None) -> None:
        """Test that cleaner handles edge cases correctly."""
        cleaner = create_text_cleaner_via_di(
            do_rm_time_schedule=False,
            do_rm_only_numeric=False,
            do_rm_include_url_text=False,
            do_rm_include_email_text=False,
        )

        result = cleaner.clean(text)
        assert result == expected
