# Tokenizer Scripts

English / [日本語](README_JA.md)

This directory contains scripts related to tokenizer training, extension, and merging.

## Scripts

### add_tokens.py

Script for adding new tokens to existing tokenizers.

#### Features
- Supports addition of normal and special tokens
- Loads token configurations from JSON files
- Can save extended tokenizers locally or push to Hugging Face Hub

#### Usage
```bash
python scripts/constract_llm/tokenizer/add_tokens.py config/constract_llm/tokenizer/add_tokens/config/config.json
```

#### Example Configuration
```json
{
    "tokenizer_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "normal_tokens_config_path": "./config/constract_llm/tokenizer/add_tokens/config/normal_tokens.json",
    "special_tokens_config_path": "./config/constract_llm/tokenizer/add_tokens/config/special_tokens.json",
    "push_to_hub": true,
    "private": true,
    "output_dir": "./data/misc/json"
}
```

#### Parameter Descriptions
| Parameter | Description |
|------------|------|
| `tokenizer_name_or_path` | Path to the tokenizer model or name of the Hugging Face model. |
| `normal_tokens_config_path` | Path to the JSON file containing normal tokens. |
| `special_tokens_config_path` | Path to the JSON file containing special tokens. |
| `push_to_hub` | Whether to push the tokenizer to the Hugging Face Hub. |
| `private` | Whether to make the model private on the Hugging Face Hub. |
| `output_dir` | Directory to save the extended tokenizer. |

### merge_spm.py

Script for merging multiple SentencePiece models.

#### Features
- Specify base tokenizer and additional tokenizer
- Can save merged models locally or push to Hugging Face Hub

#### Usage
```bash
python scripts/constract_llm/tokenizer/merge_spm.py config/constract_llm/tokenizer/merge_spm/config.json
```

#### Example Configuration
```json
{
    "base_tokenizer_name_or_path": "elyza/ELYZA-japanese-Llama-2-7b",
    "additional_tokenizer_name_or_path": "./data/misc/json",
    "output_dir": "./data/misc/json",
    "push_to_hub": true,
    "private": true
}
```

#### Parameter Descriptions
| Parameter | Description |
|------------|------|
| `base_tokenizer_name_or_path` | Directory or name of the base HF tokenizer |
| `additional_tokenizer_name_or_path` | Directory or name of the additional HF tokenizer |
| `output_dir` | Directory to save merged SPM model and HF tokenizer |
| `push_to_hub` | Push to Hugging Face Hub? |
| `private` | Hub repo private by default? |

### train_tokenizer.py

Script for training new tokenizers.

#### Features
- Supports Unigram, BPE, WordPiece model types
- Configuration of special tokens
- Options for training on large corpora
- Advanced options such as byte fallback, digit splitting, etc.

#### Usage
```bash
python scripts/constract_llm/tokenizer/train_tokenizer.py config/constract_llm/tokenizer/train_tokenizer/config.json
```

#### Example Configuration
```json
{
    "dataset_name_or_path": "wikimedia/wikipedia",
    "dataset_config": "20231101.ja",
    "split": "train",
    "text_column": "text",
    "vocab_size": 30000,
    "min_frequency": 2,
    "model_type": "unigram",
    "push_to_hub": true,
    "private": true,
    "output_dir": "./data/misc/json",
    "max_train_samples": 10000
}
```

#### Parameter Descriptions
| Parameter | Description |
|------------|------|
| `dataset_name_or_path` | HF dataset identifier, e.g. 'wikipedia' |
| `dataset_config` | Optional dataset config name |
| `split` | Split to use, e.g. 'train' |
| `text_column` | Column name for text |
| `vocab_size` | Vocabulary size |
| `model_type` | SentencePiece model type ('unigram', 'bpe', 'wordpiece') |
| `special_tokens_config` | Path to JSON of special tokens |
| `default_special_tokens` | Default special tokens |
| `max_train_samples` | Maximum number of examples to use; default is all |
| `train_extremely_large_corpus` | Enable training on extremely large corpus |
| `character_coverage` | Amount of characters covered by the model (0.0~1.0) |
| `num_threads` | Number of threads for training |
| `byte_fallback` | Enable byte fallback |
| `split_digits` | Split digits into separate tokens |
| `allow_whitespace_only_pieces` | Allow pieces containing only whitespace |
| `remove_extra_whitespaces` | Remove extra whitespaces in input |
| `input_sentence_size` | Maximum number of sentences to use for training |
| `push_to_hub` | Push to HF Hub? |
| `private` | Hub repo private? |
| `output_dir` | Directory to save tokenizer |
