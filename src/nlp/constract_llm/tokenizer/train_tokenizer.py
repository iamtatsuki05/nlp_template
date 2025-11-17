import logging
from pathlib import Path
from typing import Final, Literal

import sentencepiece as spm
from datasets import load_dataset
from transformers import LlamaTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UNK_SYMBOL: Final = '<unk>'
PAD_SYMBOL: Final = '<pad>'
BOS_SYMBOL: Final = '<s>'
EOS_SYMBOL: Final = '</s>'


def train_tokenizer(  # noqa: PLR0913
    dataset_name_or_path: str,
    output_dir: str | Path,
    *,
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
    input_sentence_size: int = 1_000_000_000,
    push_to_hub: bool = False,
    private: bool = True,
) -> None:
    logger.info(
        'Training SPM tokenizer (%s) on %s (config=%s, split=%s, max_train_samples=%s)',
        model_type,
        dataset_name_or_path,
        dataset_config,
        split,
        max_train_samples,
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
    with tmp_file.open('w', encoding='utf-8') as file_obj:
        for sample in datasets:
            if max_train_samples is not None and count >= max_train_samples:
                break
            raw_text = sample.get(text_column, sample.get('text', ''))
            text_value = str(raw_text)
            file_obj.write(text_value.replace('\n', ' ') + '\n')
            count += 1

    default_control = [PAD_SYMBOL, BOS_SYMBOL, EOS_SYMBOL]
    control_symbols = [tok for tok in (special_tokens or []) if tok in default_control]
    user_symbols = [tok for tok in (special_tokens or []) if tok not in [*default_control, UNK_SYMBOL]]

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
    logger.info('Saved SentencePiece model to %s.model', model_prefix)
    hf = LlamaTokenizer(
        vocab_file=f'{model_prefix}.model',
        unk_token=UNK_SYMBOL,
        pad_token=PAD_SYMBOL,
        bos_token=BOS_SYMBOL,
        eos_token=EOS_SYMBOL,
        extra_ids=0,
    )

    sample_text = 'こんにちは私はハチワレです。'
    enc = hf.encode(sample_text)
    dec = hf.decode(enc)

    logger.info('Text: %s', sample_text)
    logger.info('Encoding: %s', enc)
    logger.info('Decoding: %s', dec)

    hf.save_pretrained(str(output_dir))
    logger.info('Saved HF tokenizer to %s', output_dir)

    if push_to_hub:
        hf.push_to_hub(output_dir.name, private=private)
        logger.info('Pushed to Hub repo %s (private=%s)', output_dir.name, private)
