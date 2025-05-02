import logging
from pathlib import Path
from typing import Any, Literal

from nlp.common.utils.file.json import load_json, save_as_indented_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def _sequential_split(
    data: list[dict[str, Any]], n_train: int, n_val: int, n_test: int
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]] | None,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Sequentially split data into train, (optional) val, test without shuffling.
    """
    train = data[:n_train]
    val = data[n_train : n_train + n_val] if n_val else None
    test = data[n_train + n_val : n_train + n_val + n_test]
    leftover = data[n_train + n_val + n_test :]
    if leftover:
        logger.warning(f'{len(leftover)} examples leftover after split, appending to test set')
        test.extend(leftover)
    return train, val, test


def _random_split(
    data: list[dict[str, Any]], n_train: int, n_val: int, n_test: int, seed: int
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]] | None,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Randomly shuffle and split data into train, (optional) val, test.
    """
    import random

    rng = random.Random(seed)
    shuffled = data.copy()
    rng.shuffle(shuffled)
    return _sequential_split(shuffled, n_train, n_val, n_test)


def _stratified_split(
    data: list[dict[str, Any]],
    n_train: int,
    n_val: int,
    n_test: int,
    key: str,
    mode: Literal['random', 'sequential'],
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]] | None,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Stratified split by grouping on `key`, then splitting each group proportionally.
    """
    groups: dict[Any, list[dict[str, Any]]] = {}
    for item in data:
        groups.setdefault(item.get(key), []).append(item)

    train_set, val_set, test_set = [], [], []
    total = len(data)

    for grp in groups.values():
        if mode == 'random':
            import random

            rng = random.Random(seed)
            rng.shuffle(grp)
        size = len(grp)
        # proportions based on absolute counts
        g_train = int(size * (n_train / total))
        g_val = int(size * (n_val / total)) if n_val else 0
        g_test = int(size * (n_test / total))
        train_set.extend(grp[:g_train])
        if n_val:
            val_set.extend(grp[g_train : g_train + g_val])
        test_set.extend(grp[g_train + g_val : g_train + g_val + g_test])
        # any leftover goes to test
        leftover = grp[g_train + g_val + g_test :]
        if leftover:
            test_set.extend(leftover)

    if not n_val:
        val_set = None
    return train_set, val_set, test_set


def split_dataset(
    dataset_name_or_path: str | Path,
    output_dir: str | Path,
    test_size: float | int,
    val_size: float | int = None,
    split_mode: Literal['random', 'sequential'] = 'random',
    random_seed: int = 42,
    stratify_key: str | None = None,
) -> None:
    """
    Split a dataset into train/(optional) validation/test sets and save to JSON files.

    Args:
        test_size: Ratio (0<test_size<1) or absolute count for test split.
        val_size: Optional ratio or count for validation split.
                    If not provided, only train and test splits are created.
    """
    if split_mode not in ('random', 'sequential'):
        raise ValueError("split_mode must be 'random' or 'sequential'")

    # Load data
    path = Path(dataset_name_or_path)
    if path.exists():
        logger.info(f'Loading local dataset from {path}')
        data: list[dict[str, Any]] = load_json(path)
    else:
        if load_dataset is None:
            raise ImportError('datasets library required')
        logger.info(f"Loading Hugging Face dataset '{dataset_name_or_path}'")
        ds = load_dataset(str(dataset_name_or_path))
        split = 'train' if 'train' in ds else next(iter(ds))
        data = [dict(ex) for ex in ds[split]]
        logger.info(f'Loaded {len(data)} examples from split "{split}"')

    n_total = len(data)

    # compute counts
    def get_count(size: Any, name: str) -> int:
        if isinstance(size, float):
            if not 0 < size < 1:
                raise ValueError(f'{name} ratio must be between 0 and 1')
            return int(n_total * size)
        if isinstance(size, int):
            if not 0 <= size <= n_total:
                raise ValueError(f'{name} count must be between 0 and total')
            return size
        raise TypeError(f'{name} must be int or float')

    n_test = get_count(test_size, 'test_size')
    n_val = get_count(val_size, 'val_size') if val_size is not None else 0
    n_train = n_total - n_val - n_test
    if n_train < 0:
        raise ValueError('Sum of test_size and val_size exceeds total examples')

    # perform split
    if stratify_key:
        logger.info(f"Stratified {split_mode} split on '{stratify_key}'")
        train_set, val_set, test_set = _stratified_split(
            data, n_train, n_val, n_test, stratify_key, split_mode, random_seed
        )
    else:
        logger.info(f'{split_mode.capitalize()} split without stratification')
        if split_mode == 'random':
            train_set, val_set, test_set = _random_split(data, n_train, n_val, n_test, random_seed)
        else:
            train_set, val_set, test_set = _sequential_split(data, n_train, n_val, n_test)

    # save splits
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    splits = {'train.json': train_set, 'test.json': test_set}
    if n_val:
        splits['validation.json'] = val_set  # type: ignore

    for fname, items in splits.items():
        p = outdir / fname
        logger.info(f'Saving {fname} ({len(items)} examples)')
        save_as_indented_json(items, p)

    logger.info(f'Dataset split complete. Files at {outdir}')
