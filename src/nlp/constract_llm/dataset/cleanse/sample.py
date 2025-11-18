from typing import Any

from nlp.constract_llm.dataset.cleanse.text import TextCleaner, create_text_cleaner


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
    cleaner = text_cleaner or create_text_cleaner(
        do_rm_time_schedule=do_rm_time_schedule,
        rm_time_schedule_threshold=rm_time_schedule_threshold,
        do_rm_only_numeric=do_rm_only_numeric,
        do_rm_include_url_text=do_rm_include_url_text,
        do_rm_include_email_text=do_rm_include_email_text,
    )

    for col in target_cols:
        sample[col] = cleaner.clean(sample.get(col))
    return sample
