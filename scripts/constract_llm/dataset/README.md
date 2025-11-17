# Dataset Processing Scripts

English / [日本語](README_JA.md)

This directory contains scripts related to dataset processing for language model construction.

## Directory Structure

- `cleanse/` – Rules-based cleansing and deduplication utilities.
- `preprocess/` – Field-level preprocessing helpers.
- `split/` – Train/validation/test splitting with stratification support.
- `hard_negative_mine/` – Hard negative mining for retrieval-style training data.

## Processing Flow

1. **Cleansing (`cleanse/`)** – Filter duplicates, schedules, URLs, etc., and keep only clean records.
2. **Preprocessing (`preprocess/`)** – Apply lightweight normalisation to selected text fields.
3. **Splitting (`split/`)** – Produce train/validation/test partitions with reproducible seeds.
4. **Hard negative mining (`hard_negative_mine/`, optional)** – Generate mined negatives for contrastive or retrieval training.

Example commands:

```bash
python scripts/constract_llm/dataset/cleanse/cleanse.py config/constract_llm/dataset/cleanse/config.json
python scripts/constract_llm/dataset/preprocess/preprocess.py config/constract_llm/dataset/preprocess/config.json
python scripts/constract_llm/dataset/split/split.py config/constract_llm/dataset/split/config.json
python scripts/constract_llm/dataset/hard_negative_mine/hard_negative_mine.py config/custom/hard_negative.json
```

See each subdirectory’s README for configuration details and advanced options.
