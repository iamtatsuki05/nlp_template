from pathlib import Path

import fire
from pydantic import BaseModel, Field

from nlp.common.utils.cli_utils import load_cli_config
from nlp.constract_llm.dataset.cleanse.cleanse import cleanse_datasets


class CLIConfig(BaseModel):
    input_name_or_path: str = Field(..., description='Path or name of the input JSON file or Hugging Face dataset')
    output_dir: Path | str = Field(..., description='Directory where cleaned data will be saved')
    text_fields: list[str] | None = Field(None, description='list of field names to apply text cleaning')
    do_deduplicate: bool = Field(default=True, description='Whether to remove duplicate records at the record level')
    do_rm_duplicated_by_minhash: bool = Field(
        default=False, description='Whether to remove near-duplicate text entries using MinHash'
    )
    minhash_threshold: float = Field(default=0.95, description='Threshold for MinHash near-duplicate detection')
    minhash_num_perm: int = Field(default=128, description='Number of permutations for MinHash')
    num_workers: int | None = Field(default=None, description='Number of workers for parallel processing')
    do_rm_time_schedule: bool = Field(default=True, description='Remove texts containing time schedules if True')
    rm_time_schedule_threshold: int = Field(
        default=3, description='Minimum occurrences of time pattern to remove text'
    )
    do_rm_only_numeric: bool = Field(default=True, description='Remove texts that are only numeric if True')
    do_rm_include_url_text: bool = Field(default=True, description='Remove texts containing URLs if True')
    do_rm_include_email_text: bool = Field(default=True, description='Remove texts containing email addresses if True')
    max_use_samples: int | None = Field(default=None, description='Maximum number of samples to use from the dataset')
    max_save_samples: int | None = Field(
        default=None, description='Maximum number of samples to save to the output file'
    )


def main(config_file_path: str | Path, **kwargs: object) -> None:
    cfg = CLIConfig(**load_cli_config(config_file_path, **kwargs))
    cleanse_datasets(
        input_name_or_path=cfg.input_name_or_path,
        output_dir=cfg.output_dir,
        text_fields=cfg.text_fields,
        do_deduplicate=cfg.do_deduplicate,
        do_rm_duplicated_by_minhash=cfg.do_rm_duplicated_by_minhash,
        minhash_threshold=cfg.minhash_threshold,
        minhash_num_perm=cfg.minhash_num_perm,
        num_workers=cfg.num_workers,
        do_rm_time_schedule=cfg.do_rm_time_schedule,
        rm_time_schedule_threshold=cfg.rm_time_schedule_threshold,
        do_rm_only_numeric=cfg.do_rm_only_numeric,
        do_rm_include_url_text=cfg.do_rm_include_url_text,
        do_rm_include_email_text=cfg.do_rm_include_email_text,
        max_use_samples=cfg.max_use_samples,
        max_save_samples=cfg.max_save_samples,
    )


if __name__ == '__main__':
    fire.Fire(main)
