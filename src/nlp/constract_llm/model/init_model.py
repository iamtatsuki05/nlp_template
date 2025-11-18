import logging
from pathlib import Path
from typing import Any, cast

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    set_seed,
)

from nlp.constract_llm.model.model_factory import (
    ModelClassProvider,
    ModelType,
    TransformersModelClassRegistry,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def initialize_model(  # noqa: PLR0913
    model_name_or_path: str | Path,
    *,
    model_type: ModelType = 'generic',
    output_dir: str | Path | None = None,
    push_to_hub: bool = False,
    private: bool = False,
    seed: int | None = None,
    model_provider: ModelClassProvider | None = None,
) -> None:
    """Initialize a model with random weights and save it.

    Args:
        model_name_or_path: Name or path of the model to initialize.
        model_type: Type of model to initialize. Defaults to 'generic'.
        output_dir: Directory to save the initialized model. If None, uses '{model_name}-init'.
        push_to_hub: Whether to push the model to Hugging Face Hub.
        private: Whether the repository should be private when pushing to Hub.
        seed: Random seed for reproducibility.
        model_provider: Custom model class provider. If None, uses TransformersModelClassRegistry.

    """
    if seed is not None:
        set_seed(seed)

    model_name_or_path = str(model_name_or_path)
    base = Path(model_name_or_path).name
    output_dir_path = Path(output_dir) if output_dir else Path(f'{base}-init')
    output_dir_path.mkdir(parents=True, exist_ok=True)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    hf_config: PretrainedConfig = AutoConfig.from_pretrained(model_name_or_path)

    provider = model_provider or TransformersModelClassRegistry()
    model_class = provider.get_model_class(model_type)
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
