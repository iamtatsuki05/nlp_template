import logging
from pathlib import Path

import fire
from pydantic import BaseModel, Field, field_validator

from nlp.common.utils.cli_utils import load_cli_config
from nlp.constract_llm.model.init_model import initialize_model
from nlp.constract_llm.model.model_factory import ModelType, TransformersModelClassRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VALID_MODEL_TYPES = TransformersModelClassRegistry.get_supported_types()


class CLIConfig(BaseModel):
    model_name_or_path: str | Path = Field(
        ...,
        description='Path to the model or model name from Hugging Face Hub.',
    )
    model_type: ModelType = Field(
        ...,
        description='Type of the model. Choose from: ' + ', '.join(VALID_MODEL_TYPES),
    )
    output_dir: str | Path | None = Field(
        default=None,
        description='Directory to save the model.',
    )
    push_to_hub: bool = Field(
        default=False,
        description='Whether to push the model to Hugging Face Hub.',
    )
    private: bool = Field(
        default=True,
        description='Whether to make the model private on Hugging Face Hub.',
    )
    seed: int = Field(
        default=42,
        description='Random seed for initialization.',
    )

    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        if v not in VALID_MODEL_TYPES:
            raise ValueError(f"Invalid model_type '{v}'. Choose from {VALID_MODEL_TYPES}.")
        return v


def main(config_file_path: str | Path | None = None, **kwargs: object) -> None:
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
