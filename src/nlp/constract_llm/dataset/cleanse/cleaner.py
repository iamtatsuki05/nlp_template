from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from nlp.constract_llm.dataset.cleanse.rules import (
    EmailRule,
    OnlyNumericRule,
    TextRule,
    TimeScheduleRule,
    UrlRule,
)


class TextCleaner:
    def __init__(self, rules: Sequence[TextRule]) -> None:
        self._rules: tuple[TextRule, ...] = tuple(rules)

    def clean(self, text: str | None) -> str | None:
        if text is None:
            return None

        stripped = text.strip()
        if not stripped:
            return None

        for rule in self._rules:
            if rule.should_remove(stripped):
                return None

        return stripped


def create_text_cleaner(
    *,
    do_rm_time_schedule: bool = True,
    rm_time_schedule_threshold: int = 3,
    do_rm_only_numeric: bool = True,
    do_rm_include_url_text: bool = True,
    do_rm_include_email_text: bool = True,
) -> TextCleaner:
    rules: list[TextRule] = []
    if do_rm_only_numeric:
        rules.append(OnlyNumericRule())
    if do_rm_time_schedule:
        rules.append(TimeScheduleRule(threshold=rm_time_schedule_threshold))
    if do_rm_include_url_text:
        rules.append(UrlRule())
    if do_rm_include_email_text:
        rules.append(EmailRule())
    return TextCleaner(rules)


def cleanse_text(  # noqa: PLR0913
    text: str | None,
    do_rm_time_schedule: bool = True,
    rm_time_schedule_threshold: int = 3,
    do_rm_only_numeric: bool = True,
    do_rm_include_url_text: bool = True,
    do_rm_include_email_text: bool = True,
    text_cleaner: TextCleaner | None = None,
) -> str | None:
    """Cleanse a single text string by applying text cleaning rules.

    Args:
        text: Text to cleanse.
        do_rm_time_schedule: Whether to remove texts containing time schedules.
        rm_time_schedule_threshold: Minimum time patterns to trigger removal.
        do_rm_only_numeric: Whether to remove numeric-only texts.
        do_rm_include_url_text: Whether to remove texts with URLs.
        do_rm_include_email_text: Whether to remove texts with email addresses.
        text_cleaner: Optional pre-configured TextCleaner instance. If None, creates one from parameters.

    Returns:
        Cleansed text or None if the text should be removed.

    """
    cleaner = text_cleaner or create_text_cleaner(
        do_rm_time_schedule=do_rm_time_schedule,
        rm_time_schedule_threshold=rm_time_schedule_threshold,
        do_rm_only_numeric=do_rm_only_numeric,
        do_rm_include_url_text=do_rm_include_url_text,
        do_rm_include_email_text=do_rm_include_email_text,
    )
    return cleaner.clean(text)
