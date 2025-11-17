import logging
from pathlib import Path
from typing import Any, Final, cast

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
from transformers.models.auto.auto_factory import _BaseAutoModelClass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VALID_MODEL_TYPES: Final = [
    'seq2seq',
    'causal',
    'masked',
    'generic',
]


def compute_param_stats(model: torch.nn.Module) -> dict[str, int]:
    def stats(params: list[torch.nn.Parameter]) -> tuple[float, float, int]:
        if not params:
            return float('nan'), float('nan'), 0
        flat = torch.cat([p.view(-1) for p in params if p.requires_grad])
        return flat.mean().item(), flat.std().item(), sum(p.numel() for p in params if p.requires_grad)

    all_params = [p for p in model.parameters() if p.requires_grad]
    all_mean, all_std, all_count = stats(all_params)

    non_emb_params = [p for n, p in model.named_parameters() if p.requires_grad and 'embed' not in n.lower()]
    non_mean, non_std, non_count = stats(non_emb_params)

    logger.info('[ALL PARAMS] mean=%s, std=%s, count=%d', all_mean, all_std, all_count)
    logger.info('[NON-EMBED PARAMS] mean=%s, std=%s, count=%d', non_mean, non_std, non_count)

    return {
        'all_params': all_count,
        'non_embedding_params': non_count,
    }


def load_model_class(model_type: str) -> type[_BaseAutoModelClass]:
    match model_type:
        case 'seq2seq':
            return AutoModelForSeq2SeqLM
        case 'causal':
            return AutoModelForCausalLM
        case 'masked':
            return AutoModelForMaskedLM
        case 'generic':
            return AutoModel
        case _:
            raise ValueError(f"Invalid model_type '{model_type}'. Choose from {VALID_MODEL_TYPES}.")


def initialize_model(  # noqa: PLR0913
    model_name_or_path: str | Path,
    *,
    model_type: str = 'generic',
    output_dir: str | Path | None = None,
    push_to_hub: bool = False,
    private: bool = False,
    seed: int | None = None,
) -> None:
    if seed is not None:
        set_seed(seed)

    model_name_or_path = str(model_name_or_path)
    base = Path(model_name_or_path).name
    output_dir_path = Path(output_dir) if output_dir else Path(f'{base}-init')
    output_dir_path.mkdir(parents=True, exist_ok=True)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    hf_config: PretrainedConfig = AutoConfig.from_pretrained(model_name_or_path)

    model_class: type[_BaseAutoModelClass] = load_model_class(model_type)
    model: PreTrainedModel = cast('PreTrainedModel', model_class.from_config(hf_config))
    logger.info("%s model '%s' initialized with random weights.", model_type, model_name_or_path)

    compute_param_stats(model)

    model.save_pretrained(output_dir_path)
    tokenizer.save_pretrained(output_dir_path)
    logger.info("Model and tokenizer saved to '%s'.", output_dir_path)

    if push_to_hub:
        repo_id = output_dir_path.name
        logger.info("Pushing to Hugging Face Hub: repo='%s', private=%s", repo_id, private)
        hub_model: Any = model
        hub_model.push_to_hub(repo_id, private=private)
        tokenizer.push_to_hub(repo_id, private=private)
        logger.info('Push to Hub completed.')
