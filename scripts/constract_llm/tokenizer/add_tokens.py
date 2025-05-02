from pathlib import Path
from typing import Any

import fire
from pydantic import BaseModel, Field

from nlp.common.utils.cli_utils import load_cli_config
from nlp.constract_llm.tokenizer.add_tokens import extend_and_save_tokenizer


class CLIConfig(BaseModel):
    tokenizer_name_or_path: str | Path = Field(
        ...,
        description='Path to the tokenizer model or name of the Hugging Face model.',
    )
    normal_tokens_config_path: str | Path = Field(
        ...,
        description='Path to the JSON file containing normal tokens.',
    )
    special_tokens_config_path: str | Path = Field(
        ...,
        description='Path to the JSON file containing special tokens.',
    )
    push_to_hub: bool = Field(
        False,
        description='Whether to push the tokenizer to the Hugging Face Hub.',
    )
    private: bool = Field(
        True,
        description='Whether to make the model private on the Hugging Face Hub.',
    )
    output_dir: str | Path = Field(
        ...,
        description='Directory to save the extended tokenizer.',
    )


def main(config_file_path: str | Path, **kwargs: Any) -> None:
    cfg = CLIConfig(**load_cli_config(config_file_path, **kwargs))

    extend_and_save_tokenizer(
        tokenizer_name_or_path=cfg.tokenizer_name_or_path,
        normal_tokens_file_path=cfg.normal_tokens_config_path,
        special_tokens_file_path=cfg.special_tokens_config_path,
        output_dir=cfg.output_dir,
        push_to_hub=cfg.push_to_hub,
        private=cfg.private,
    )


if __name__ == '__main__':
    fire.Fire(main)
