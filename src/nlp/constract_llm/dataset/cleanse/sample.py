from typing import Any

from nlp.constract_llm.dataset.cleanse.di import create_text_cleaner_via_di
from nlp.constract_llm.dataset.cleanse.text import TextCleaner


def cleanse_sample(  # noqa: PLR0913
    sample: dict[str, Any],
    target_cols: list[str],
    do_rm_time_schedule: bool = True,
    rm_time_schedule_threshold: int = 3,
    do_rm_only_numeric: bool = True,
    do_rm_include_url_text: bool = True,
    do_rm_include_email_text: bool = True,
    text_cleaner: TextCleaner | None = None,
) -> dict[str, Any]:
    """Cleanse a single sample by applying text cleaning rules to target columns.

    Args:
        sample: Dictionary containing the data to cleanse.
        target_cols: List of column names to apply cleaning rules to.
        do_rm_time_schedule: Whether to remove texts containing time schedules.
        rm_time_schedule_threshold: Minimum time patterns to trigger removal.
        do_rm_only_numeric: Whether to remove numeric-only texts.
        do_rm_include_url_text: Whether to remove texts with URLs.
        do_rm_include_email_text: Whether to remove texts with email addresses.
        text_cleaner: Optional pre-configured TextCleaner instance. If None, creates one from parameters.

    Returns:
        Cleansed sample dictionary with updated values.

    """
    cleaner = text_cleaner or create_text_cleaner_via_di(
        do_rm_time_schedule=do_rm_time_schedule,
        rm_time_schedule_threshold=rm_time_schedule_threshold,
        do_rm_only_numeric=do_rm_only_numeric,
        do_rm_include_url_text=do_rm_include_url_text,
        do_rm_include_email_text=do_rm_include_email_text,
    )

    for col in target_cols:
        sample[col] = cleaner.clean(sample.get(col))
    return sample
