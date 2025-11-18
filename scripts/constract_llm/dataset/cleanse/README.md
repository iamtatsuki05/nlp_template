# Dataset Cleansing Script

English / [日本語](README_JA.md)

This directory contains scripts for dataset cleansing.

## Overview

The `cleanse.py` script provides various cleansing operations to remove unwanted data from raw datasets and create high-quality datasets.

## Features

- Removal of duplicate data
- Detection and removal of similar texts using MinHash
- Removal of texts containing time schedules
- Removal of texts containing only numbers
- Removal of texts containing URLs or email addresses

## Usage

```bash
python scripts/constract_llm/dataset/cleanse/cleanse.py config/constract_llm/dataset/cleanse/config.json
```

You can also override configuration file values using command-line arguments:

```bash
python scripts/constract_llm/dataset/cleanse/cleanse.py config/constract_llm/dataset/cleanse/config.json --do_deduplicate=False
```

## Configuration File

The configuration file is located at `config/constract_llm/dataset/cleanse/config.json`.

### Example Configuration

```json
{
    "input_name_or_path": "imdb",
    "output_dir": "./data/misc/json",
    "text_fields": ["text"],
    "do_deduplicate": true,
    "do_rm_duplicated_by_minhash": true,
    "minhash_threshold": 0.95,
    "do_rm_time_schedule": true,
    "rm_time_schedule_threshold": 3,
    "do_rm_only_numeric": true,
    "do_rm_include_url_text": true,
    "do_rm_include_email_text": true
}
```

### Parameter Descriptions

| Parameter                     | Description                                                 |
| ----------------------------- | ----------------------------------------------------------- |
| `input_name_or_path`          | Path or name of the input JSON file or Hugging Face dataset |
| `output_dir`                  | Directory where cleaned data will be saved                  |
| `text_fields`                 | List of field names to apply text cleaning                  |
| `do_deduplicate`              | Whether to remove duplicate records at the record level     |
| `do_rm_duplicated_by_minhash` | Whether to remove near-duplicate text entries using MinHash |
| `minhash_threshold`           | Threshold for MinHash near-duplicate detection (0.0-1.0)    |
| `minhash_num_perm`            | Number of permutations for MinHash                          |
| `num_workers`                 | Number of workers for parallel processing                   |
| `do_rm_time_schedule`         | Whether to remove texts containing time schedules           |
| `rm_time_schedule_threshold`  | Minimum occurrences of time pattern to remove text          |
| `do_rm_only_numeric`          | Whether to remove texts that are only numeric               |
| `do_rm_include_url_text`      | Whether to remove texts containing URLs                     |
| `do_rm_include_email_text`    | Whether to remove texts containing email addresses          |
| `max_use_samples`             | Maximum number of samples to use from the dataset           |
| `max_save_samples`            | Maximum number of samples to save to the output file        |
