import logging
from pathlib import Path
from typing import Literal

import fire
from pydantic import BaseModel, Field

from nlp.common.utils.cli_utils import load_cli_config
from nlp.constract_llm.dataset.split.split import split_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIConfig(BaseModel):
    dataset_name_or_path: str = Field(
        ...,
        description='Path to the input dataset. Can be a local file or a Hugging Face dataset name.',
    )
    output_dir: str = Field(
        ...,
        description='Directory to save the processed data.',
    )
    test_size: float | int = Field(
        default=0.2,
        description=(
            'Proportion of the dataset to include in the test split (float) or absolute number of test samples (int).'
        ),
    )
    val_size: float | int | None = Field(
        default=None,
        description=(
            'Proportion of the dataset to include in the validation split (float) '
            'or absolute number of validation samples (int).'
        ),
    )
    split_mode: Literal['random', 'sequential'] = Field(
        default='random',
        description='Mode of splitting the dataset. Can be "random" or "sequential".',
    )
    random_seed: int = Field(
        default=42,
        description='Random seed for reproducibility.',
    )
    stratify_key: str | None = Field(
        default=None,
        description='Key to stratify the split. If None, no stratification is applied.',
    )


def main(config_file_path: str | Path, **kwargs: object) -> None:
    cfg = CLIConfig(**load_cli_config(config_file_path, **kwargs))
    split_dataset(
        dataset_name_or_path=cfg.dataset_name_or_path,
        output_dir=cfg.output_dir,
        test_size=float(cfg.test_size),
        val_size=float(cfg.val_size) if cfg.val_size is not None else None,
        split_mode=cfg.split_mode,
        random_seed=cfg.random_seed,
        stratify_key=cfg.stratify_key,
    )


if __name__ == '__main__':
    fire.Fire(main)
