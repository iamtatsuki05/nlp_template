import logging
import random
from pathlib import Path
from typing import Any, Literal, SupportsFloat

from datasets import load_dataset
from tqdm.auto import tqdm

from nlp.common.utils.file.json import load_json, save_as_indented_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _sequential_split(
    data: list[dict[str, Any]],
    n_train: int,
    n_val: int,
    n_test: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """Sequentially split data into train, (optional) val, test without shuffling."""
    train = data[:n_train]
    val = data[n_train : n_train + n_val] if n_val else None
    test = data[n_train + n_val : n_train + n_val + n_test]
    leftover = data[n_train + n_val + n_test :]
    if leftover:
        logger.warning('%s examples leftover after split, appending to test set', len(leftover))
        test.extend(leftover)
    return train, val, test


def _random_split(
    data: list[dict[str, Any]],
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """Randomly shuffle and split data into train, (optional) val, test."""
    rng = random.Random(seed)  # deterministic shuffling for reproducibility  # noqa: S311
    shuffled = data.copy()
    rng.shuffle(shuffled)
    return _sequential_split(shuffled, n_train, n_val, n_test)


def _stratified_split(  # noqa: PLR0913
    data: list[dict[str, Any]],
    n_train: int,
    n_val: int,
    n_test: int,
    key: str,
    mode: Literal['random', 'sequential'],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """Stratified split by grouping on `key`, then splitting each group proportionally."""
    groups: dict[Any, list[dict[str, Any]]] = {}
    for item in data:
        groups.setdefault(item.get(key), []).append(item)

    train_set: list[dict[str, Any]] = []
    val_set: list[dict[str, Any]] | None = [] if n_val else None
    test_set: list[dict[str, Any]] = []
    total = len(data)

    for grp in tqdm(groups.values(), desc='Stratified split', unit='group'):
        if mode == 'random':
            rng = random.Random(seed)  # noqa: S311
            rng.shuffle(grp)
        size = len(grp)
        g_train = int(size * (n_train / total)) if total else 0
        g_val = int(size * (n_val / total)) if total and n_val else 0
        g_test = int(size * (n_test / total)) if total else 0
        train_set.extend(grp[:g_train])
        if n_val:
            if val_set is None:
                msg = 'Validation set buffer not initialized.'
                raise ValueError(msg)
            val_set.extend(grp[g_train : g_train + g_val])
        test_set.extend(grp[g_train + g_val : g_train + g_val + g_test])
        leftover = grp[g_train + g_val + g_test :]
        if leftover:
            test_set.extend(leftover)

    if not n_val:
        val_set = None
    return train_set, val_set, test_set


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
    split_mode: Literal['random', 'sequential'] = 'random',
    random_seed: int = 42,
    stratify_key: str | None = None,
) -> None:
    """Split a dataset into train/(optional) validation/test sets and save to JSON files."""
    path = Path(dataset_name_or_path)
    if path.exists():
        logger.info('Loading local dataset from %s', path)
        data = load_json(path)
        if not isinstance(data, list):
            raise TypeError(f'Local dataset {path} must be a list of records.')
        records = [dict(record) for record in data]
    else:
        logger.info("Loading Hugging Face dataset '%s'", dataset_name_or_path)
        ds = load_dataset(str(dataset_name_or_path))
        split = 'train' if 'train' in ds else next(iter(ds))
        records = [dict(ex) for ex in ds[split]]
        logger.info('Loaded %s examples from split "%s"', len(records), split)

    n_total = len(records)
    n_test = _resolve_count(test_size, 'test_size', n_total)
    n_val = _resolve_count(val_size, 'val_size', n_total) if val_size is not None else 0
    n_train = n_total - n_val - n_test
    if n_train < 0:
        raise ValueError('Sum of test_size and val_size exceeds total examples')

    if stratify_key:
        train_set, val_set, test_set = _stratified_split(
            data=records,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            key=stratify_key,
            mode=split_mode,
            seed=random_seed,
        )
    elif split_mode == 'random':
        train_set, val_set, test_set = _random_split(
            data=records,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            seed=random_seed,
        )
    else:
        train_set, val_set, test_set = _sequential_split(
            data=records,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
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
