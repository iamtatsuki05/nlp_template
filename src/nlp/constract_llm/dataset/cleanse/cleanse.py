import logging
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from nlp.common.utils.file.io import save_file
from nlp.constract_llm.dataset.cleanse.cleaner import TextCleaner
from nlp.constract_llm.dataset.cleanse.di import create_text_cleaner_via_di
from nlp.constract_llm.dataset.cleanse.duplicates import cleanse_column_duplicates
from nlp.constract_llm.dataset.cleanse.sample import cleanse_sample
from nlp.constract_llm.dataset.loader import load_dataset_resource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cleanse_datasets(  # noqa: PLR0913, C901
    input_name_or_path: str,
    output_dir: Path | str,
    text_fields: list[str] | None = None,
    do_deduplicate: bool = True,
    do_rm_duplicated_by_minhash: bool = False,
    minhash_threshold: float = 0.95,
    minhash_num_perm: int = 128,
    num_workers: int | None = None,
    do_rm_time_schedule: bool = True,
    rm_time_schedule_threshold: int = 3,
    do_rm_only_numeric: bool = True,
    do_rm_include_url_text: bool = True,
    do_rm_include_email_text: bool = True,
    max_use_samples: int | None = None,
    max_save_samples: int | None = None,
    text_cleaner: TextCleaner | None = None,
) -> None:
    """Cleanse datasets by removing duplicates and applying text cleaning rules.

    Args:
        input_name_or_path: Dataset name or path to load from.
        output_dir: Directory to save cleaned datasets.
        text_fields: List of text field names to clean.
        do_deduplicate: Whether to remove duplicate records.
        do_rm_duplicated_by_minhash: Whether to use MinHash for near-duplicate detection.
        minhash_threshold: Similarity threshold for MinHash (0.0-1.0).
        minhash_num_perm: Number of permutations for MinHash.
        num_workers: Number of worker processes for parallel processing.
        do_rm_time_schedule: Whether to remove texts containing time schedules.
        rm_time_schedule_threshold: Minimum time patterns to trigger removal.
        do_rm_only_numeric: Whether to remove numeric-only texts.
        do_rm_include_url_text: Whether to remove texts with URLs.
        do_rm_include_email_text: Whether to remove texts with email addresses.
        max_use_samples: Maximum samples to use from input (before cleaning).
        max_save_samples: Maximum samples to save (after cleaning).
        text_cleaner: Optional pre-configured TextCleaner instance. If None, creates one from parameters.

    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    def clean_and_save_split(dataset: list[dict[str, Any]], split_name: str) -> None:
        if max_use_samples is not None and len(dataset) > max_use_samples:
            dataset = dataset[:max_use_samples]

        # Deduplicate the dataset
        if do_deduplicate:
            seen, unique_records = set(), []
            for item in tqdm(dataset, desc=f"Deduplicating split '{split_name}'", unit='rec'):
                key = tuple(item.get(field_name, '') for field_name in (text_fields or item.keys()))
                if key not in seen:
                    seen.add(key)
                    unique_records.append(item)
            dataset = unique_records

        if text_fields:
            # Cleanse for each text field
            if do_deduplicate:
                # Remove near-duplicate texts using MinHash
                for field in text_fields:
                    cleaned_texts, removed = cleanse_column_duplicates(
                        dataset,
                        field,
                        do_rm_duplicated_by_minhash=do_rm_duplicated_by_minhash,
                        threshold=minhash_threshold,
                        num_perm=minhash_num_perm,
                        num_workers=num_workers,
                    )
                    dataset = [
                        {**record, field: new_text}
                        for record, new_text in zip(dataset, cleaned_texts, strict=True)
                        if new_text is not None
                    ]
                    logger.info(f"Removed {removed} near-duplicate entries in field '{field}' (split: '{split_name}')")

            cleaner = text_cleaner or create_text_cleaner_via_di(
                do_rm_time_schedule=do_rm_time_schedule,
                rm_time_schedule_threshold=rm_time_schedule_threshold,
                do_rm_only_numeric=do_rm_only_numeric,
                do_rm_include_url_text=do_rm_include_url_text,
                do_rm_include_email_text=do_rm_include_email_text,
            )

            # Remove rule-based logic
            dataset = [
                cleaned
                for raw in tqdm(dataset, desc=f"Sample cleaning split '{split_name}'", unit='rec')
                if (
                    cleaned := cleanse_sample(
                        raw,
                        text_fields,
                        text_cleaner=cleaner,
                    )
                )
                and any(cleaned.get(field) is not None for field in text_fields)
            ]

            logger.info(f"Cleaned {len(dataset)} records in split '{split_name}'")

        if max_save_samples is not None and len(dataset) > max_save_samples:
            dataset = dataset[:max_save_samples]

        filename = f'{split_name}.json' if split_name else 'cleansed.json'
        save_file(dataset, outdir / filename)
        logger.info(f'Saved {len(dataset)} records to {outdir / filename}')

    dataset_resource = load_dataset_resource(
        input_name_or_path,
        local_split_name='',
        allow_remote_fallback=True,
    )
    for split_name, split_dataset in dataset_resource.iter_splits():
        logger.info("Processing split '%s' (%s records)", split_name or 'default', len(split_dataset))
        clean_and_save_split(split_dataset, split_name)
