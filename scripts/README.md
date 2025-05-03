# Scripts

English / [日本語](README_JA.md)

This directory contains various scripts used in the NLP template project. These scripts are primarily designed to automate tasks related to building and training language models (LLMs).

## Basic Scripts

- `main.py` - A sample script that displays a basic "Hello, World!".

## constract_llm

Contains a series of scripts related to language model construction.

### dataset

Scripts related to dataset processing.

#### cleanse

- `cleanse.py` - Script for dataset cleansing.
  - Removal of duplicate data
  - Detection and removal of similar texts using MinHash
  - Removal of texts containing time schedules
  - Removal of texts containing only numbers
  - Removal of texts containing URLs or email addresses
  - Various cleansing options can be customized using configuration files

#### preprocess

- `preprocess.py` - Script for dataset preprocessing.
  - Applies preprocessing to specified text fields
  - Input can be a local file or Hugging Face dataset name
  - Saves processed data to the specified directory

#### split

- `split.py` - Script for splitting datasets into training, validation, and test sets.
  - Supports random or sequential splitting
  - Test and validation sizes can be specified as ratios or absolute numbers
  - Supports stratified sampling (based on a specified key)
  - Random seed setting for reproducibility

### model

- `init_model.py` - Script for model initialization.
  - Supports various model types
  - Loads models from Hugging Face Hub or local paths
  - Can save models locally or push to Hugging Face Hub
  - Seed setting for reproducibility

### tokenizer

Scripts related to tokenizers.

- `add_tokens.py` - Script for adding new tokens to existing tokenizers.
  - Supports addition of normal and special tokens
  - Loads token configurations from JSON files
  - Can save extended tokenizers locally or push to Hugging Face Hub

- `merge_spm.py` - Script for merging multiple SentencePiece models.
  - Specify base tokenizer and additional tokenizer
  - Can save merged models locally or push to Hugging Face Hub

- `train_tokenizer.py` - Script for training new tokenizers.
  - Supports Unigram, BPE, WordPiece model types
  - Configuration of special tokens
  - Options for training on large corpora
  - Advanced options such as byte fallback, digit splitting, etc.
  - Can save trained tokenizers locally or push to Hugging Face Hub

## Usage

Each script provides a CLI interface using [Google Fire](https://github.com/google/python-fire). Basic usage is as follows:

```bash
python scripts/constract_llm/dataset/cleanse/cleanse.py config/constract_llm/dataset/cleanse/config.json
```

Each script accepts a corresponding configuration file and executes processing based on that configuration. Configuration files can be provided in JSON, YAML, TOML, or other formats.

You can also override configuration file values using command-line arguments:

```bash
python scripts/constract_llm/dataset/cleanse/cleanse.py config/constract_llm/dataset/cleanse/config.json --do_deduplicate=False
```

## Configuration Files

Configuration files for each script are stored in the `config/` directory. For example:

- Dataset cleansing: `config/constract_llm/dataset/cleanse/config.json`
- Dataset preprocessing: `config/constract_llm/dataset/preprocess/config.json`
- Dataset splitting: `config/constract_llm/dataset/split/config.json`
- Model initialization: `config/constract_llm/model/init_model/config.json`
- Tokenizer training: `config/constract_llm/tokenizer/train_tokenizer/config.json`

Each configuration file contains all parameters used by the corresponding script.
