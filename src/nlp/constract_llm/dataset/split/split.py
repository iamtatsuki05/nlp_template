import logging
from pathlib import Path
from typing import Any, SupportsFloat

from nlp.common.utils.file.json import save_as_indented_json
from nlp.constract_llm.dataset.loader import load_dataset_resource
from nlp.constract_llm.dataset.split.di import create_split_strategy
from nlp.constract_llm.dataset.split.strategy import SplitMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _resolve_count(size: SupportsFloat, name: str, total: int) -> int:
    if isinstance(size, float):
        if not 0 < size < 1:
            raise ValueError(f'{name} ratio must be between 0 and 1')
        return int(total * size)
    if isinstance(size, int):
        if not 0 <= size <= total:
            raise ValueError(f'{name} count must be between 0 and total')
        return size
    raise TypeError(f'{name} must be int or float')


def split_dataset(  # noqa: PLR0913
    dataset_name_or_path: str | Path,
    output_dir: str | Path,
    test_size: SupportsFloat,
    val_size: SupportsFloat | None = None,
    split_mode: SplitMode = 'random',
    random_seed: int = 42,
    stratify_key: str | None = None,
) -> None:
    """Split a dataset into train/(optional) validation/test sets and save to JSON files.

    Args:
        dataset_name_or_path: Path to local JSON or HuggingFace dataset name
        output_dir: Directory where split files will be saved
        test_size: Test set size (float for ratio, int for count)
        val_size: Validation set size (float for ratio, int for count), optional
        split_mode: Splitting strategy ('random', 'sequential', or 'stratified')
        random_seed: Random seed for reproducible splits
        stratify_key: Field name to stratify by (required if split_mode='stratified')

    Raises:
        ValueError: If split sizes are invalid or stratify_key is missing for stratified mode

    """
    logger.info('Loading dataset from %s', dataset_name_or_path)
    dataset = load_dataset_resource(dataset_name_or_path)
    preferred_split = 'train' if dataset.has_split('train') else None
    split_name, records = dataset.pick_split(preferred_split)
    if dataset.is_local:
        logger.info('Loaded local dataset: %s records', len(records))
    else:
        logger.info('Loaded %s examples from split "%s"', len(records), split_name)

    n_total = len(records)
    n_test = _resolve_count(test_size, 'test_size', n_total)
    n_val = _resolve_count(val_size, 'val_size', n_total) if val_size is not None else 0
    n_train = n_total - n_val - n_test
    if n_train < 0:
        raise ValueError('Sum of test_size and val_size exceeds total examples')

    # Create split strategy via dependency injection
    # If stratified mode requested, ensure stratify_key is provided
    effective_mode = split_mode
    if split_mode == 'stratified' and not stratify_key:
        logger.warning('stratified mode requested but no stratify_key provided, falling back to random')
        effective_mode = 'random'

    strategy = create_split_strategy(
        mode=effective_mode,
        random_seed=random_seed,
        stratify_key=stratify_key,
    )

    # Perform the split using the strategy
    train_set, val_set, test_set = strategy.split(
        data=records,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
    )

    # Log if any leftover examples
    actual_total = len(train_set) + len(test_set) + (len(val_set) if val_set else 0)
    if actual_total > n_total:
        logger.warning(
            '%s examples leftover after split, appended to test set',
            actual_total - n_total,
        )

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[dict[str, Any]]] = {'train.json': train_set, 'test.json': test_set}
    if n_val:
        if val_set is None:
            msg = 'Validation set is missing despite val_size being specified.'
            raise ValueError(msg)
        splits['validation.json'] = val_set

    for fname, items in splits.items():
        target = outdir / fname
        logger.info('Saving %s (%s examples)', fname, len(items))
        save_as_indented_json(items, target)

    logger.info('Dataset split complete. Files at %s', outdir)
