# Dataset Splitting Script

English / [日本語](README_JA.md)

This directory contains scripts for splitting datasets into training, validation, and test sets.

## Overview

The `split.py` script splits datasets into training, validation, and test sets, which is an important step for training and evaluating machine learning models.

## Features

- Supports random or sequential splitting
- Test and validation sizes can be specified as ratios or absolute numbers
- Supports stratified sampling (based on a specified key)
- Random seed setting for reproducibility

## Usage

```bash
python scripts/constract_llm/dataset/split/split.py config/constract_llm/dataset/split/config.json
```

You can also override configuration file values using command-line arguments:

```bash
python scripts/constract_llm/dataset/split/split.py config/constract_llm/dataset/split/config.json --test_size=0.2 --val_size=0.1
```

## Configuration File

The configuration file is located at `config/constract_llm/dataset/split/config.json`.

### Example Configuration

```json
{
    "dataset_name_or_path": "imdb",
    "output_dir": "./data/misc/json",
    "test_size": 0.1,
    "val_size": 0.1,
    "split_mode": "random",
    "random_seed": 42,
    "stratify_key": "label"
}
```

### Parameter Descriptions

| Parameter | Description |
|------------|------|
| `dataset_name_or_path` | Path to the input dataset. Can be a local file or a Hugging Face dataset name. |
| `output_dir` | Directory to save the split data. |
| `test_size` | Proportion of the dataset to include in the test split (float) or absolute number of test samples (int). |
| `val_size` | Proportion of the dataset to include in the validation split (float) or absolute number of validation samples (int). |
| `split_mode` | Mode of splitting the dataset. Can be "random" or "sequential". |
| `random_seed` | Random seed for reproducibility. |
| `stratify_key` | Key to stratify the split. If None, no stratification is applied. |
