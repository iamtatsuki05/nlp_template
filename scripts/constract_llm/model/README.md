# Model Initialization Script

English / [日本語](README_JA.md)

This directory contains scripts for model initialization.

## Overview

The `init_model.py` script initializes pre-trained models and prepares them for subsequent fine-tuning or other processing.

## Features

- Supports various model types
- Loads models from Hugging Face Hub or local paths
- Can save models locally or push to Hugging Face Hub
- Seed setting for reproducibility

## Usage

```bash
python scripts/constract_llm/model/init_model.py config/constract_llm/model/init_model/config.json
```

You can also override configuration file values using command-line arguments:

```bash
python scripts/constract_llm/model/init_model.py config/constract_llm/model/init_model/config.json --model_type=encoder --push_to_hub=False
```

## Configuration File

The configuration file is located at `config/constract_llm/model/init_model/config.json`.

### Example Configuration

```json
{
  "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "output_dir": "./data/misc/json",
  "model_type": "causal",
  "push_to_hub": true,
  "private": true,
  "seed": 42
}
```

### Parameter Descriptions

| Parameter | Description |
|------------|------|
| `model_name_or_path` | Path to the model or model name from Hugging Face Hub. |
| `model_type` | Type of the model (`seq2seq`, `causal`, `masked`, `generic`). |
| `output_dir` | Directory to save the model. |
| `push_to_hub` | Whether to push the model to Hugging Face Hub. |
| `private` | Whether to make the model private on Hugging Face Hub. |
| `seed` | Random seed for initialization. |
