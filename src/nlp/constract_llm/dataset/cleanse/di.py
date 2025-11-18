"""Dependency injection module for text cleansing."""

from __future__ import annotations

from injector import Injector, Module, provider, singleton

from nlp.constract_llm.dataset.cleanse.text import (
    EmailRule,
    OnlyNumericRule,
    TextCleaner,
    TextRule,
    TimeScheduleRule,
    UrlRule,
)


class TextCleanerModule(Module):
    """Injector module for TextCleaner configuration.

    This module provides a flexible way to configure TextCleaner with
    different combinations of cleaning rules through dependency injection.

    Examples:
        >>> module = TextCleanerModule(
        ...     do_rm_only_numeric=True,
        ...     do_rm_include_url_text=True,
        ... )
        >>> injector = Injector([module])
        >>> cleaner = injector.get(TextCleaner)
        >>> result = cleaner.clean("123.45")
        >>> assert result is None

    """

    def __init__(
        self,
        *,
        do_rm_time_schedule: bool = True,
        rm_time_schedule_threshold: int = 3,
        do_rm_only_numeric: bool = True,
        do_rm_include_url_text: bool = True,
        do_rm_include_email_text: bool = True,
    ) -> None:
        """Initialize the module with text cleaning configuration.

        Args:
            do_rm_time_schedule: Whether to remove texts containing time schedules.
            rm_time_schedule_threshold: Minimum number of time patterns to trigger removal.
            do_rm_only_numeric: Whether to remove texts that are only numeric.
            do_rm_include_url_text: Whether to remove texts containing URLs.
            do_rm_include_email_text: Whether to remove texts containing email addresses.

        """
        self._do_rm_time_schedule = do_rm_time_schedule
        self._rm_time_schedule_threshold = rm_time_schedule_threshold
        self._do_rm_only_numeric = do_rm_only_numeric
        self._do_rm_include_url_text = do_rm_include_url_text
        self._do_rm_include_email_text = do_rm_include_email_text

    @singleton
    @provider
    def text_cleaner(self) -> TextCleaner:
        """Provide configured TextCleaner instance.

        Returns:
            TextCleaner configured with the rules specified in __init__.

        """
        rules: list[TextRule] = []
        if self._do_rm_only_numeric:
            rules.append(OnlyNumericRule())
        if self._do_rm_time_schedule:
            rules.append(TimeScheduleRule(threshold=self._rm_time_schedule_threshold))
        if self._do_rm_include_url_text:
            rules.append(UrlRule())
        if self._do_rm_include_email_text:
            rules.append(EmailRule())
        return TextCleaner(rules)


def create_text_cleaner_via_di(
    *,
    do_rm_time_schedule: bool = True,
    rm_time_schedule_threshold: int = 3,
    do_rm_only_numeric: bool = True,
    do_rm_include_url_text: bool = True,
    do_rm_include_email_text: bool = True,
) -> TextCleaner:
    """Create TextCleaner via dependency injection.

    This function provides a convenient way to create a TextCleaner instance
    using dependency injection, which makes testing easier by allowing mock
    rules to be injected.

    Args:
        do_rm_time_schedule: Whether to remove texts containing time schedules.
        rm_time_schedule_threshold: Minimum number of time patterns to trigger removal.
        do_rm_only_numeric: Whether to remove texts that are only numeric.
        do_rm_include_url_text: Whether to remove texts containing URLs.
        do_rm_include_email_text: Whether to remove texts containing email addresses.

    Returns:
        TextCleaner configured with the specified rules.

    Examples:
        >>> cleaner = create_text_cleaner_via_di(
        ...     do_rm_only_numeric=True,
        ...     do_rm_include_url_text=False,
        ... )
        >>> result = cleaner.clean("123")
        >>> assert result is None

    """
    module = TextCleanerModule(
        do_rm_time_schedule=do_rm_time_schedule,
        rm_time_schedule_threshold=rm_time_schedule_threshold,
        do_rm_only_numeric=do_rm_only_numeric,
        do_rm_include_url_text=do_rm_include_url_text,
        do_rm_include_email_text=do_rm_include_email_text,
    )
    injector = Injector([module])
    return injector.get(TextCleaner)
