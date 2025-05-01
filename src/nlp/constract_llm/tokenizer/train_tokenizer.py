import logging
from pathlib import Path
from typing import Literal

import sentencepiece as spm
from datasets import load_dataset
from transformers import LlamaTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_tokenizer(
    dataset_name_or_path: str,
    output_dir: str | Path,
    dataset_config: str | None = None,
    split: str = 'train',
    text_column: str = 'text',
    vocab_size: int = 30000,
    model_type: Literal['unigram', 'bpe', 'word', 'char'] = 'unigram',
    special_tokens: list[str] | None = None,
    max_train_samples: int | None = None,
    train_extremely_large_corpus: bool = False,
    character_coverage: float = 1.0,
    num_threads: int = -1,
    byte_fallback: bool = True,
    split_digits: bool = True,
    allow_whitespace_only_pieces: bool = True,
    remove_extra_whitespaces: bool = False,
    input_sentence_size: int = 1000000000,
    push_to_hub: bool = False,
    private: bool = True,
) -> None:
    logger.info(
        f'Training SPM tokenizer ({model_type}) on {dataset_name_or_path} '
        f'(config={dataset_config}, split={split}, max_train_samples={max_train_samples})'
    )

    datasets = (
        load_dataset(dataset_name_or_path, dataset_config, split=split, streaming=True)
        if dataset_config
        else load_dataset(dataset_name_or_path, split=split, streaming=True)
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = output_dir / 'spm_input.txt'
    count = 0
    with tmp_file.open('w', encoding='utf-8') as f:
        for sample in datasets:
            if max_train_samples is not None and count >= max_train_samples:
                break
            text = sample.get(text_column, sample.get('text', ''))
            f.write(text.replace('\n', ' ') + '\n')
            count += 1
    default_control = ['<pad>', '<s>', '</s>']
    control_symbols = [tok for tok in (special_tokens or []) if tok in default_control]
    user_symbols = [tok for tok in (special_tokens or []) if tok not in default_control + ['<unk>']]

    model_prefix = str(output_dir / 'spm_tokenizer')
    spm.SentencePieceTrainer.train(
        input=str(tmp_file),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        train_extremely_large_corpus=train_extremely_large_corpus,
        num_threads=num_threads,
        byte_fallback=byte_fallback,
        split_digits=split_digits,
        allow_whitespace_only_pieces=allow_whitespace_only_pieces,
        remove_extra_whitespaces=remove_extra_whitespaces,
        input_sentence_size=input_sentence_size,
        control_symbols=control_symbols or None,
        user_defined_symbols=user_symbols or None,
    )
    logger.info(f'Saved SentencePiece model to {model_prefix}.model')

    if push_to_hub:
        hf = LlamaTokenizer(
            vocab_file=f'{model_prefix}.model',
            unk_token='<unk>',
            pad_token='<pad>',
            bos_token='<s>',
            eos_token='</s>',
            extra_ids=0,
        )
        hf.push_to_hub(output_dir.name, private=private)

        text = 'こんにちは私はハチワレです。'
        enc = hf.encode(text)
        dec = hf.decode(enc)

        logger.info(f'Text: {text}')
        logger.info(f'Encoding: {enc}')
        logger.info(f'Decoding: {dec}')

        logger.info(f'Pushed to Hub repo {output_dir.name} (private={private})')
