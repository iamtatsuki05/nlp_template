import logging
import unicodedata
from pathlib import Path
from typing import Any

from nlp.common.utils.file.json import save_as_indented_json
from nlp.constract_llm.dataset.loader import load_dataset_resource

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
    dataset = load_dataset_resource(input_name_or_path)
    preferred_split = 'train' if dataset.has_split('train') else None
    split_name, data = dataset.pick_split(preferred_split)
    if dataset.is_local:
        logger.info('Loaded local JSON: %s records', len(data))
    else:
        logger.info(
            "Loaded HF dataset '%s': %s records from split '%s'",
            dataset.source,
            len(data),
            split_name,
        )

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
