from pathlib import Path
from typing import Any

import fire
from pydantic import BaseModel, Field

from nlp.common.utils.cli_utils import load_cli_config
from nlp.constract_llm.dataset.preprocess.preprocess import preprocess_data


class CLIConfig(BaseModel):
    input_name_or_path: str | Path = Field(
        ...,
        description='Path to the input dataset. Can be a local file or a Hugging Face dataset name.',
    )
    output_dir: str | Path = Field(
        ...,
        description='Directory to save the processed data.',
    )
    text_fields: list[str] = Field(
        default_factory=list,
        description='List of text fields to process. If not provided, all fields will be processed.',
    )


def main(config_file_path: str | Path, **kwargs: Any) -> None:
    cfg = CLIConfig(**load_cli_config(config_file_path, **kwargs))
    preprocess_data(
        input_name_or_path=cfg.input_name_or_path,
        output_dir=cfg.output_dir,
        text_fields=cfg.text_fields,
    )


if __name__ == '__main__':
    fire.Fire(main)
