import logging
from pathlib import Path
from typing import Final, Type

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    set_seed,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VALID_MODEL_TYPES: Final = [
    'seq2seq',
    'causal',
    'masked',
    'generic',
]


def compute_param_stats(model: torch.nn.Module) -> int:
    params = torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad])
    mean_val = params.mean().item()
    std_val = params.std().item()
    total_params = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)

    logger.debug(f'weights mean={mean_val:.4f}, std={std_val:.4f}')
    logger.info(f'Total trainable parameters: {total_params}')
    return total_params


def load_model_class(model_type: str) -> Type[PreTrainedModel]:
    match model_type:
        case 'seq2seq':
            return AutoModelForSeq2SeqLM  # type: ignore[return-value]
        case 'causal':
            return AutoModelForCausalLM  # type: ignore[return-value]
        case 'masked':
            return AutoModelForMaskedLM  # type: ignore[return-value]
        case 'generic':
            return AutoModel  # type: ignore[return-value]
        case _:
            raise ValueError(f"Invalid model_type '{model_type}'. Choose from {VALID_MODEL_TYPES}.")


def initialize_model(
    model_name_or_path: str,
    model_type: str = 'generic',
    output_dir: str | Path | None = None,
    push_to_hub: bool = False,
    private: bool = False,
    seed: int | None = None,
) -> None:
    """
    Initialize a model with random weights and save it to the specified directory.
    Optionally push the model to the Hugging Face Hub.
    """
    if seed is not None:
        set_seed(seed)

    base = model_name_or_path.split('/')[-1]
    output_dir = Path(output_dir) if output_dir else Path(f'{base}-init')
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    config: PretrainedConfig = AutoConfig.from_pretrained(model_name_or_path)

    model_class: Type[PreTrainedModel] = load_model_class(model_type)
    model: PreTrainedModel = model_class.from_config(config)
    logger.info(f"{model_type} model '{model_name_or_path}' initialized with random weights.")

    compute_param_stats(model)

    model.save_pretrained(output_dir)  # type: ignore[attr-defined]
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model and tokenizer saved to '{output_dir}'.")

    if push_to_hub:
        repo_id = output_dir.name
        logger.info(f"Pushing to Hugging Face Hub: repo='{repo_id}', private={private}")
        model.push_to_hub(repo_id, private=private)  # type: ignore[attr-defined]
        tokenizer.push_to_hub(repo_id, private=private)
        logger.info('Push to Hub completed.')
