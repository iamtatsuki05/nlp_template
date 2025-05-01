import logging
from pathlib import Path
from typing import Any

import fire
from pydantic import BaseModel, field_validator

from nlp.common.utils.cli_utils import load_cli_config
from nlp.constract_llm.model.init_model import VALID_MODEL_TYPES, initialize_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIConfig(BaseModel):
    model_name_or_path: str
    model_type: str = 'generic'
    output_dir: str | Path | None = None
    push_to_hub: bool = False
    private: bool = False
    seed: int | None = None

    @field_validator('model_type')
    def validate_model_type(cls, v: str) -> str:
        if v not in VALID_MODEL_TYPES:
            raise ValueError(f"Invalid model_type '{v}'. Choose from {VALID_MODEL_TYPES}.")
        return v


def main(config_file_path: str | Path | None = None, **kwargs: Any) -> None:
    cfg = CLIConfig(**load_cli_config(config_file_path, **kwargs))
    initialize_model(
        model_name_or_path=cfg.model_name_or_path,
        model_type=cfg.model_type,
        output_dir=cfg.output_dir,
        push_to_hub=cfg.push_to_hub,
        private=cfg.private,
        seed=cfg.seed,
    )
    logger.info('Model initialization completed.')


if __name__ == '__main__':
    fire.Fire(main)
