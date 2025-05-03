# Dataset Preprocessing Script

English / [日本語](README_JA.md)

This directory contains scripts for dataset preprocessing.

## Overview

The `preprocess.py` script applies basic preprocessing to datasets, preparing them for subsequent processing steps.

## Features

- Applies preprocessing to specified text fields
- Input can be a local file or Hugging Face dataset name
- Saves processed data to the specified directory

## Usage

```bash
python scripts/constract_llm/dataset/preprocess/preprocess.py config/constract_llm/dataset/preprocess/config.json
```

You can also override configuration file values using command-line arguments:

```bash
python scripts/constract_llm/dataset/preprocess/preprocess.py config/constract_llm/dataset/preprocess/config.json --text_fields='["content"]'
```

## Configuration File

The configuration file is located at `config/constract_llm/dataset/preprocess/config.json`.

### Example Configuration

```json
{
    "input_name_or_path": "imdb",
    "output_dir": "./data/misc/json",
    "text_fields": ["text", "label"]
}
```

### Parameter Descriptions

| Parameter | Description |
|------------|------|
| `input_name_or_path` | Path to the input dataset. Can be a local file or a Hugging Face dataset name. |
| `output_dir` | Directory to save the processed data. |
| `text_fields` | List of text fields to process. If not provided, all fields will be processed. |
