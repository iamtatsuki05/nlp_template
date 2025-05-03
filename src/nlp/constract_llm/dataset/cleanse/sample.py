from typing import Any

from nlp.constract_llm.dataset.cleanse.text import cleanse_text


def cleanse_sample(
    sample: dict[str, Any],
    target_cols: list[str],
    do_rm_time_schedule: bool = True,
    rm_time_schedule_threshold: int = 3,
    do_rm_only_numeric: bool = True,
    do_rm_include_url_text: bool = True,
    do_rm_include_email_text: bool = True,
) -> dict[str, Any]:
    for col in target_cols:
        original_text = sample.get(col)
        cleansed_text = cleanse_text(
            original_text,
            do_rm_time_schedule=do_rm_time_schedule,
            rm_time_schedule_threshold=rm_time_schedule_threshold,
            do_rm_only_numeric=do_rm_only_numeric,
            do_rm_include_url_text=do_rm_include_url_text,
            do_rm_include_email_text=do_rm_include_email_text,
        )
        sample[col] = cleansed_text
    return sample
