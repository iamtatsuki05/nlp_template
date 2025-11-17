import logging
import unicodedata
from pathlib import Path
from typing import Any

from datasets import load_dataset

from nlp.common.utils.file.json import load_json, save_as_indented_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Unicode normalization and whitespace collapsing."""
    text = unicodedata.normalize('NFKC', text)
    return ' '.join(text.split())


def preprocess_data(
    input_name_or_path: str | Path,
    output_dir: str | Path,
    text_fields: list[str] | None = None,
) -> None:
    """Load cleansed JSON or HF dataset pre-cleansed, apply text normalization, save preprocessed data."""
    path = Path(input_name_or_path)
    if path.exists():
        loaded = load_json(path)
        if not isinstance(loaded, list):
            msg = f'Expected list records at {path}, but found {type(loaded).__name__}'
            raise TypeError(msg)
        data = [dict(item) for item in loaded]
        logger.info('Loaded local JSON: %s records', len(data))
    else:
        ds = load_dataset(input_name_or_path)
        split = 'train' if 'train' in ds else next(iter(ds))
        data = [dict(ex) for ex in ds[split]]
        logger.info(f"Loaded HF dataset '{input_name_or_path}': {len(data)} records from split '{split}'")

    processed: list[dict[str, Any]] = []
    for item in data:
        new_item = item.copy()
        for k, v in item.items():
            if isinstance(v, str) and (text_fields is None or k in text_fields):
                new_item[k] = clean_text(v)
        processed.append(new_item)

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / 'preprocessed.json'
    save_as_indented_json(processed, out_path)
    logger.info(f'Preprocessed data saved: {len(processed)} records -> {out_path}')
