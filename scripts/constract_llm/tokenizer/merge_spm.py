from pathlib import Path
from typing import Any

import fire
from pydantic import BaseModel, Field

from nlp.common.utils.cli_utils import load_cli_config
from nlp.constract_llm.tokenizer.merge_spm import merge_spm_models


class CLIConfig(BaseModel):
    base_tokenizer_name_or_path: str | Path = Field(..., description='Directory or name of the base HF tokenizer')
    additional_tokenizer_name_or_path: str | Path = Field(
        ..., description='Directory or name of the additional HF tokenizer'
    )
    output_dir: str | Path = Field(..., description='Directory to save merged SPM model and HF tokenizer')
    push_to_hub: bool = Field(False, description='Push to Hugging Face Hub?')
    private: bool = Field(True, description='Hub repo private by default?')


def main(config_file_path: str | Path, **kwargs: Any) -> None:
    cfg = CLIConfig(**load_cli_config(config_file_path, **kwargs))

    merge_spm_models(
        base_tokenizer_name_or_path=cfg.base_tokenizer_name_or_path,
        additional_tokenizer_name_or_path=cfg.additional_tokenizer_name_or_path,
        output_dir=cfg.output_dir,
        push_to_hub=cfg.push_to_hub,
        private=cfg.private,
    )


if __name__ == '__main__':
    fire.Fire(main)
