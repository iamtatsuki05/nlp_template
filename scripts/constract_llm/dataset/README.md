# Dataset Processing Scripts

English / [日本語](README_JA.md)

This directory contains scripts related to dataset processing for language model construction.

## Directory Structure

- `cleanse/`: Scripts for dataset cleansing
- `preprocess/`: Scripts for dataset preprocessing
- `split/`: Scripts for splitting datasets into training, validation, and test sets

## Data Processing Flow

A typical data processing flow is as follows:

1. **Cleansing (cleanse)**: Remove unwanted data from raw data to create a high-quality dataset
2. **Preprocessing (preprocess)**: Apply basic preprocessing to cleansed data
3. **Splitting (split)**: Split processed data into training, validation, and test sets

## Usage Examples

Here are examples of commands for processing the IMDB dataset:

```bash
# 1. Dataset cleansing
python scripts/constract_llm/dataset/cleanse/cleanse.py config/constract_llm/dataset/cleanse/config.json

# 2. Preprocessing of cleansed data
python scripts/constract_llm/dataset/preprocess/preprocess.py config/constract_llm/dataset/preprocess/config.json

# 3. Splitting of processed data
python scripts/constract_llm/dataset/split/split.py config/constract_llm/dataset/split/config.json
```

For detailed usage instructions and configuration options for each script, please refer to the README in each directory.
