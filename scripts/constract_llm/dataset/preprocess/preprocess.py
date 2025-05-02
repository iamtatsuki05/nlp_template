from pathlib import Path
from typing import Any

import fire
from pydantic import BaseModel

from nlp.common.utils.cli_utils import load_cli_config
from nlp.constract_llm.dataset.preprocess.preprocess import preprocess_data


class CLIConfig(BaseModel):
    input_name_or_path: str
    output_dir: str | Path
    text_fields: list[str] | None = None


def main(config_file_path: str | Path, **kwargs: Any) -> None:
    cfg = CLIConfig(**load_cli_config(config_file_path, **kwargs))
    preprocess_data(
        input_name_or_path=cfg.input_name_or_path,
        output_dir=cfg.output_dir,
        text_fields=cfg.text_fields,
    )


if __name__ == '__main__':
    fire.Fire(main)
