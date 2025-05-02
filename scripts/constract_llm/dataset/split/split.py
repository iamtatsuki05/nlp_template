import logging
from pathlib import Path
from typing import Any, Literal

import fire
from pydantic import BaseModel

from nlp.common.utils.cli_utils import load_cli_config
from nlp.constract_llm.dataset.split.split import split_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIConfig(BaseModel):
    dataset_name_or_path: str | Path
    output_dir: str | Path
    test_size: float | int
    val_size: float | int = None
    split_mode: Literal['random', 'sequential'] = 'random'
    random_seed: int = 42
    stratify_key: str | None = None


def main(config_file_path: str | Path, **kwargs: Any) -> None:
    cfg = CLIConfig(**load_cli_config(config_file_path, **kwargs))
    split_dataset(
        cfg.dataset_name_or_path,
        cfg.output_dir,
        cfg.test_size,
        cfg.val_size,
        cfg.split_mode,
        cfg.random_seed,
        cfg.stratify_key,
    )


if __name__ == '__main__':
    fire.Fire(main)
