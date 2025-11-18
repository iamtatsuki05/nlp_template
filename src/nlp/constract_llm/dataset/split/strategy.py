"""Dataset splitting strategies using Protocol and ABC for extensibility.

This module provides different strategies for splitting datasets into train/val/test sets.
Each strategy implements the SplitStrategy protocol, allowing easy extension and testing.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, Protocol

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from collections.abc import Sequence

# Global type definition for split modes
SplitMode = Literal['sequential', 'random', 'stratified']


class SplitStrategy(Protocol):
    """Protocol defining the interface for dataset splitting strategies.

    Implementations should split data into train, optional validation, and test sets
    according to specified counts.
    """

    def split(
        self,
        data: Sequence[dict[str, Any]],
        n_train: int,
        n_val: int,
        n_test: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, list[dict[str, Any]]]:
        """Split data into train, validation (optional), and test sets.

        Args:
            data: Dataset records to split
            n_train: Number of training examples
            n_val: Number of validation examples (0 means no validation set)
            n_test: Number of test examples

        Returns:
            Tuple of (train_set, val_set, test_set).
            val_set is None if n_val is 0.
            Any leftover examples are appended to test_set.

        """
        ...


class BaseSplitter(ABC):
    """Abstract base class providing common splitting logic."""

    @abstractmethod
    def split(
        self,
        data: Sequence[dict[str, Any]],
        n_train: int,
        n_val: int,
        n_test: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, list[dict[str, Any]]]:
        """Split data into train, validation, and test sets."""

    @staticmethod
    def _sequential_partition(
        data: Sequence[dict[str, Any]],
        n_train: int,
        n_val: int,
        n_test: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, list[dict[str, Any]]]:
        """Partition data sequentially without shuffling.

        This is a helper method used by concrete strategies.
        """
        data_list = list(data)
        train = data_list[:n_train]
        val = data_list[n_train : n_train + n_val] if n_val else None
        test = data_list[n_train + n_val : n_train + n_val + n_test]
        leftover = data_list[n_train + n_val + n_test :]
        if leftover:
            test.extend(leftover)
        return train, val, test


class SequentialSplitter(BaseSplitter):
    """Split data sequentially without shuffling.

    Examples are taken in order: first n_train for training,
    next n_val for validation, remaining for test.
    """

    def split(
        self,
        data: Sequence[dict[str, Any]],
        n_train: int,
        n_val: int,
        n_test: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, list[dict[str, Any]]]:
        """Split data sequentially without shuffling."""
        return self._sequential_partition(data, n_train, n_val, n_test)


class RandomSplitter(BaseSplitter):
    """Split data with random shuffling for better distribution.

    Args:
        seed: Random seed for reproducibility

    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def split(
        self,
        data: Sequence[dict[str, Any]],
        n_train: int,
        n_val: int,
        n_test: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, list[dict[str, Any]]]:
        """Randomly shuffle data before splitting."""
        rng = random.Random(self.seed)  # noqa: S311
        shuffled = list(data)
        rng.shuffle(shuffled)
        return self._sequential_partition(shuffled, n_train, n_val, n_test)


class StratifiedSplitter(BaseSplitter):
    """Split data while preserving class distribution across splits.

    Groups data by a specified key and splits each group proportionally.
    Useful when you want to maintain label balance across train/val/test.

    Args:
        key: Field name to stratify by (e.g., 'label', 'category')
        mode: Whether to shuffle within groups ('random') or keep order ('sequential')
        seed: Random seed when mode is 'random'

    """

    def __init__(
        self,
        key: str,
        mode: Literal['random', 'sequential'] = 'random',
        seed: int = 42,
    ) -> None:
        self.key = key
        self.mode = mode
        self.seed = seed

    def split(
        self,
        data: Sequence[dict[str, Any]],
        n_train: int,
        n_val: int,
        n_test: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, list[dict[str, Any]]]:
        """Split data with stratification by key field."""
        # Group by stratification key
        groups: dict[Any, list[dict[str, Any]]] = {}
        for item in data:
            groups.setdefault(item.get(self.key), []).append(item)

        # Initialize result containers
        train_set: list[dict[str, Any]] = []
        val_set: list[dict[str, Any]] | None = [] if n_val else None
        test_set: list[dict[str, Any]] = []
        total = len(data)

        # Split each group proportionally
        for grp in tqdm(groups.values(), desc='Stratified split', unit='group'):
            if self.mode == 'random':
                rng = random.Random(self.seed)  # noqa: S311
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

            # Append any leftover to test set
            leftover = grp[g_train + g_val + g_test :]
            if leftover:
                test_set.extend(leftover)

        if not n_val:
            val_set = None
        return train_set, val_set, test_set
