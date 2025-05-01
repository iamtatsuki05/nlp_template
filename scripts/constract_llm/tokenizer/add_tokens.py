from pathlib import Path
from typing import Any

import fire
from pydantic import BaseModel

from nlp.common.utils.cli_utils import load_cli_config
from nlp.constract_llm.tokenizer.add_tokens import extend_and_save_tokenizer


class CLIConfig(BaseModel):
    tokenizer_name_or_path: str | Path
    normal_tokens_config_path: str | Path
    special_tokens_config_path: str | Path
    push_to_hub: bool = False
    private: bool = True
    output_dir: str | Path


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
