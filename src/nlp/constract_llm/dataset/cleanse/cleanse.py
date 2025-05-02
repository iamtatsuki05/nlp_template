import logging
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm.auto import tqdm

from nlp.common.utils.file.json import load_json, save_as_indented_json
from nlp.constract_llm.dataset.cleanse.sample import cleanse_sample
from nlp.constract_llm.dataset.cleanse.text import cleanse_column_duplicates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cleanse_datasets(
    input_name_or_path: str,
    output_dir: Path | str,
    text_fields: list[str] | None = None,
    do_deduplicate: bool = True,
    do_rm_duplicated_by_minhash: bool = False,
    minhash_threshold: float = 0.95,
    do_rm_time_schedule: bool = True,
    rm_time_schedule_threshold: int = 3,
    do_rm_only_numeric: bool = True,
    do_rm_include_url_text: bool = True,
    do_rm_include_email_text: bool = True,
    max_use_samples: int | None = None,
    max_save_samples: int | None = None,
) -> None:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    def clean_and_save_split(dataset: list[dict[str, Any]], split_name: str):
        if max_use_samples is not None and len(dataset) > max_use_samples:
            dataset = dataset[:max_use_samples]

        if do_deduplicate:
            seen, unique_records = set(), []
            for item in tqdm(dataset, desc=f"Deduplicating split '{split_name}'", unit='rec'):
                key = tuple(item.get(field_name, '') for field_name in (text_fields or item.keys()))
                if key not in seen:
                    seen.add(key)
                    unique_records.append(item)
            dataset = unique_records

        if text_fields:
            if do_deduplicate:
                for field_name in text_fields:
                    cleaned_texts, removed = cleanse_column_duplicates(
                        dataset,
                        field_name,
                        do_rm_duplicated_by_minhash=do_rm_duplicated_by_minhash,
                        threshold=minhash_threshold,
                    )
                    for record, new_value in zip(dataset, cleaned_texts):
                        record[field_name] = new_value
                    logger.info(
                        f"Removed {removed} near-duplicate entries in field '{field_name}' (split: '{split_name}')"
                    )

            options = {
                'do_rm_time_schedule': do_rm_time_schedule,
                'rm_time_schedule_threshold': rm_time_schedule_threshold,
                'do_rm_only_numeric': do_rm_only_numeric,
                'do_rm_include_url_text': do_rm_include_url_text,
                'do_rm_include_email_text': do_rm_include_email_text,
            }
            dataset = [
                cleanse_sample(item, text_fields, **options)
                for item in tqdm(dataset, desc=f"Sample cleaning split '{split_name}'", unit='rec')
            ]
            logger.info(f"Cleaned {len(dataset)} records in split '{split_name}'")

        if max_save_samples is not None and len(dataset) > max_save_samples:
            dataset = dataset[:max_save_samples]

        filename = f'{split_name}.json' if split_name else 'cleansed.json'
        save_as_indented_json(dataset, outdir / filename)
        logger.info(f'Saved {len(dataset)} records to {outdir / filename}')

    try:
        dataset = load_json(Path(input_name_or_path))
        logger.info(f'Loaded local JSON: {len(dataset)} records')
        clean_and_save_split(dataset, '')
    except Exception:
        ds = load_dataset(input_name_or_path)
        for split in ds:
            split_dataset = [dict(ex) for ex in ds[split]]
            logger.info(f"Loaded split '{split}': {len(split_dataset)} records")
            clean_and_save_split(split_dataset, split)
