from pathlib import Path
from typing import Any, Literal

import fire
from pydantic import BaseModel, Field, PositiveInt, conint

from nlp.common.utils.cli_utils import load_cli_config
from nlp.common.utils.file.json import load_json
from nlp.constract_llm.tokenizer.train_tokenizer import train_tokenizer


class CLIConfig(BaseModel):
    dataset_name_or_path: str = Field(..., description="HF dataset identifier, e.g. 'wikipedia'")
    dataset_config: str | None = Field(None, description='Optional dataset config name')
    split: str = Field('train', description="Split to use, e.g. 'train'")
    text_column: str = Field('text', description='Column name for text')
    vocab_size: PositiveInt = Field(30000, description='Vocabulary size')
    model_type: Literal['unigram', 'bpe', 'wordpiece'] = Field('unigram', description='SentencePiece model type')
    special_tokens_config: str | None = Field(None, description='Path to JSON of special tokens')
    default_special_tokens: list[str] = Field(
        default_factory=lambda: ['<unk>', '<s>', '</s>', '<pad>', '<mask>', '<CLS>', '<SEP>', '<EOD>', '<MASK>', '\n'],
        description='Default special tokens',
    )
    max_train_samples: PositiveInt | None = Field(
        None,
        description='Maximum number of examples to use; default is all',
    )
    train_extremely_large_corpus: bool = Field(False, description='Enable training on extremely large corpus')
    character_coverage: float = Field(1.0, description='Amount of characters covered by the model (0.0~1.0)')
    num_threads: conint(gt=0) = Field(1, description='Number of threads for training')
    byte_fallback: bool = Field(True, description='Enable byte fallback')
    split_digits: bool = Field(True, description='Split digits into separate tokens')
    allow_whitespace_only_pieces: bool = Field(True, description='Allow pieces containing only whitespace')
    remove_extra_whitespaces: bool = Field(False, description='Remove extra whitespaces in input')
    input_sentence_size: int = Field(
        1000000000,
        description='Maximum number of sentences to use for training; default is all',
    )
    push_to_hub: bool = Field(False, description='Push to HF Hub?')
    private: bool = Field(True, description='Hub repo private?')
    output_dir: Path = Field(..., description='Directory to save tokenizer')


def main(config_file_path: str | Path, **kwargs: Any) -> None:
    cfg = CLIConfig(**load_cli_config(config_file_path, **kwargs))

    if cfg.special_tokens_config and Path(cfg.special_tokens_config).exists():
        specials = load_json(Path(cfg.special_tokens_config))
    else:
        specials = cfg.default_special_tokens

    train_tokenizer(
        dataset_name_or_path=cfg.dataset_name_or_path,
        output_dir=cfg.output_dir,
        dataset_config=cfg.dataset_config,
        split=cfg.split,
        text_column=cfg.text_column,
        vocab_size=cfg.vocab_size,
        model_type=cfg.model_type,
        special_tokens=specials,
        max_train_samples=cfg.max_train_samples,
        train_extremely_large_corpus=cfg.train_extremely_large_corpus,
        character_coverage=cfg.character_coverage,
        num_threads=cfg.num_threads,
        byte_fallback=cfg.byte_fallback,
        split_digits=cfg.split_digits,
        allow_whitespace_only_pieces=cfg.allow_whitespace_only_pieces,
        remove_extra_whitespaces=cfg.remove_extra_whitespaces,
        input_sentence_size=cfg.input_sentence_size,
        push_to_hub=cfg.push_to_hub,
        private=cfg.private,
    )


if __name__ == '__main__':
    fire.Fire(main)
