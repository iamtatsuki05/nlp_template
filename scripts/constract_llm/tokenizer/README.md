# Tokenizer Scripts

English / [日本語](README_JA.md)

This directory contains scripts related to tokenizer training, extension, and merging.

## Scripts

Extend an existing tokenizer with additional normal or special tokens.

**Key features**
- Loads lists of normal and special tokens from JSON files.
- Persists the extended tokenizer locally and can push it to the Hugging Face Hub.

**Usage**

```bash
python scripts/constract_llm/tokenizer/add_tokens.py config/constract_llm/tokenizer/add_tokens/config/config.json
```

Templates for normal/special token lists are stored under `config/constract_llm/tokenizer/add_tokens/config/`.

## `merge_spm.py`

Merge a base SentencePiece model with an additional one and export the result as both SentencePiece and Hugging Face tokenizer formats.

**Key features**
- Accepts local paths or Hub identifiers for base and additional tokenizers.
- Saves the merged artefacts locally and optionally pushes to the Hub.

**Usage**

```bash
python scripts/constract_llm/tokenizer/merge_spm.py config/constract_llm/tokenizer/merge_spm/config.json
```

## `train_tokenizer.py`

Train a new SentencePiece tokenizer from a Hugging Face dataset or local corpus.

**Key features**
- Supports `unigram`, `bpe`, and `wordpiece` model types via SentencePiece Trainer.
- Handles very large corpora with streaming and `train_extremely_large_corpus` options.
- Provides advanced switches such as byte fallback, digit splitting, and whitespace control.
- Allows custom special tokens from JSON or defaults defined in the CLI config.

**Usage**

```bash
python scripts/constract_llm/tokenizer/train_tokenizer.py config/constract_llm/tokenizer/train_tokenizer/config.json
```

**Important parameters**

| Parameter                                                                                      | Description                                                               |
| ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `dataset_name_or_path`                                                                         | HF dataset identifier or local path.                                      |
| `dataset_config`                                                                               | Optional dataset configuration name.                                      |
| `split`                                                                                        | Dataset split to load (default `train`).                                  |
| `text_column`                                                                                  | Field that contains the source text.                                      |
| `vocab_size`                                                                                   | Target vocabulary size.                                                   |
| `model_type`                                                                                   | SentencePiece model type （`unigram` / `bpe` / `word` / `char`）.                 |
| `special_tokens_config`                                                                        | JSON file that lists special tokens to add.                               |
| `default_special_tokens`                                                                       | Built-in fallback list used when no JSON is supplied.                     |
| `max_train_samples`                                                                            | Maximum number of examples to read (processes entire dataset if omitted). |
| `train_extremely_large_corpus`                                                                 | Enable SentencePiece large-corpus mode.                                   |
| `character_coverage`                                                                           | Portion of characters covered by the model (0–1).                         |
| `num_threads`                                                                                  | Number of threads used by SentencePiece Trainer.                          |
| `byte_fallback` / `split_digits` / `allow_whitespace_only_pieces` / `remove_extra_whitespaces` | Advanced preprocessing switches.                                          |
| `input_sentence_size`                                                                          | Maximum sentences consumed during training.                               |
| `push_to_hub`, `private`, `output_dir`                                                         | Output control and Hub publishing options.                                |

Configuration templates reside in `config/constract_llm/tokenizer/`.
