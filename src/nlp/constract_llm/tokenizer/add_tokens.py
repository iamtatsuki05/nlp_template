import logging
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nlp.common.utils.file.json import load_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_tokens_to_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    normal_tokens: list[str] | None = None,
    special_tokens: list[str] | None = None,
) -> PreTrainedTokenizerBase:
    if normal_tokens is not None:
        added = tokenizer.add_tokens(normal_tokens)
        logger.info(f'Added {added} normal tokens')
    if special_tokens is not None:
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        logger.info(f'Added special tokens {special_tokens}')
    return tokenizer


def _load_token_list(path: Path, kind: str) -> list[str] | None:
    data = load_json(path)
    if data is None:
        return None
    if not isinstance(data, list) or not all(isinstance(token, str) for token in data):
        raise TypeError(f'{kind} file {path} must contain a JSON list of strings.')
    return data


def extend_and_save_tokenizer(  # noqa: PLR0913
    tokenizer_name_or_path: str | Path,
    normal_tokens_file_path: str | Path,
    special_tokens_file_path: str | Path,
    output_dir: str | Path,
    push_to_hub: bool = False,
    private: bool = True,
) -> None:
    tokenizer_name_or_path = Path(tokenizer_name_or_path)
    normal_tokens_file_path = Path(normal_tokens_file_path)
    special_tokens_file_path = Path(special_tokens_file_path)

    logger.info('Loading normal tokens from %s', normal_tokens_file_path)
    normal_tokens = _load_token_list(normal_tokens_file_path, 'Normal tokens')
    logger.info('Loading special tokens from %s', special_tokens_file_path)
    special_tokens = _load_token_list(special_tokens_file_path, 'Special tokens')

    logger.info('Loading tokenizer from %s', tokenizer_name_or_path)
    if tokenizer_name_or_path.is_file():
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=str(tokenizer_name_or_path.parent),
            tokenizer_file=str(tokenizer_name_or_path),
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_name_or_path))
    added_tokenizer = add_tokens_to_tokenizer(
        tokenizer,
        normal_tokens=normal_tokens,
        special_tokens=special_tokens,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    added_tokenizer.save_pretrained(str(output_dir))
    logger.info('Saved to %s', output_dir)
    if push_to_hub:
        added_tokenizer.push_to_hub(output_dir.name, private=private)
        logger.info('Pushed to hub repo %s (private=%s)', output_dir.name, private)
