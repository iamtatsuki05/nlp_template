# Language Model Construction Scripts

English / [日本語](README_JA.md)

This directory contains a series of scripts related to the construction and training of language models (LLMs).

## Directory Structure

- `dataset/`: Scripts related to dataset processing
  - `cleanse/`: Dataset cleansing
  - `preprocess/`: Dataset preprocessing
  - `split/`: Dataset splitting
- `model/`: Scripts related to model initialization
- `tokenizer/`: Scripts related to tokenizers
  - Token addition
  - SPM model merging
  - Tokenizer training

## Language Model Construction Flow

A typical language model construction flow is as follows:

1. **Dataset Processing**:
   - Cleansing: Removal of unwanted data
   - Preprocessing: Application of basic preprocessing
   - Splitting: Division into training, validation, and test sets

2. **Tokenizer Preparation**:
   - Training a new tokenizer
   - Or adding tokens to an existing tokenizer
   - Merging multiple SPM models if necessary

3. **Model Initialization**:
   - Loading a pre-trained model
   - Or initializing a new model

4. **Model Training**:
   - Training the model using the processed dataset

## Usage Examples

Here are examples of commands for language model construction:

```bash
# 1. Dataset cleansing
python scripts/constract_llm/dataset/cleanse/cleanse.py config/constract_llm/dataset/cleanse/config.json

# 2. Preprocessing of cleansed data
python scripts/constract_llm/dataset/preprocess/preprocess.py config/constract_llm/dataset/preprocess/config.json

# 3. Splitting of processed data
python scripts/constract_llm/dataset/split/split.py config/constract_llm/dataset/split/config.json

# 4. Tokenizer training
python scripts/constract_llm/tokenizer/train_tokenizer.py config/constract_llm/tokenizer/train_tokenizer/config.json

# 5. Adding tokens to tokenizer
python scripts/constract_llm/tokenizer/add_tokens.py config/constract_llm/tokenizer/add_tokens/config/config.json

# 6. Model initialization
python scripts/constract_llm/model/init_model.py config/constract_llm/model/init_model/config.json
```

For detailed usage instructions and configuration options for each script, please refer to the README in each directory.
