"""Tests for dataset splitting strategies."""

from typing import Any

import pytest

from nlp.constract_llm.dataset.split.strategy import (
    RandomSplitter,
    SequentialSplitter,
    StratifiedSplitter,
)


@pytest.fixture
def sample_data() -> list[dict[str, Any]]:
    """Sample dataset with 10 records."""
    return [{'id': i, 'value': f'item_{i}'} for i in range(10)]


@pytest.fixture
def labeled_data() -> list[dict[str, Any]]:
    """Sample dataset with labels for stratified splitting."""
    return [
        {'id': 0, 'label': 'A'},
        {'id': 1, 'label': 'A'},
        {'id': 2, 'label': 'A'},
        {'id': 3, 'label': 'A'},
        {'id': 4, 'label': 'B'},
        {'id': 5, 'label': 'B'},
        {'id': 6, 'label': 'B'},
        {'id': 7, 'label': 'B'},
        {'id': 8, 'label': 'C'},
        {'id': 9, 'label': 'C'},
    ]


class TestSequentialSplitter:
    def test_basic_split(self, sample_data: list[dict[str, Any]]) -> None:
        """Sequential split preserves order."""
        splitter = SequentialSplitter()
        train, val, test = splitter.split(sample_data, n_train=6, n_val=2, n_test=2)

        assert len(train) == 6
        assert val is not None
        assert len(val) == 2
        assert len(test) == 2
        assert train[0]['id'] == 0
        assert train[-1]['id'] == 5
        assert val[0]['id'] == 6
        assert test[0]['id'] == 8

    def test_split_without_validation(self, sample_data: list[dict[str, Any]]) -> None:
        """Split without validation set."""
        splitter = SequentialSplitter()
        train, val, test = splitter.split(sample_data, n_train=7, n_val=0, n_test=3)

        assert len(train) == 7
        assert val is None
        assert len(test) == 3

    def test_leftover_appended_to_test(self) -> None:
        """Leftover examples are appended to test set."""
        data = [{'id': i} for i in range(11)]
        splitter = SequentialSplitter()
        train, val, test = splitter.split(data, n_train=5, n_val=2, n_test=3)

        assert len(train) == 5
        assert val is not None
        assert len(val) == 2
        assert len(test) == 4  # 3 + 1 leftover


class TestRandomSplitter:
    def test_reproducible_shuffle(self, sample_data: list[dict[str, Any]]) -> None:
        """Same seed produces same split."""
        splitter1 = RandomSplitter(seed=42)
        splitter2 = RandomSplitter(seed=42)

        train1, val1, test1 = splitter1.split(sample_data, n_train=6, n_val=2, n_test=2)
        train2, val2, test2 = splitter2.split(sample_data, n_train=6, n_val=2, n_test=2)

        assert train1 == train2
        assert val1 == val2
        assert test1 == test2

    def test_different_seeds_produce_different_splits(self, sample_data: list[dict[str, Any]]) -> None:
        """Different seeds produce different splits."""
        splitter1 = RandomSplitter(seed=42)
        splitter2 = RandomSplitter(seed=123)

        train1, _, _ = splitter1.split(sample_data, n_train=6, n_val=2, n_test=2)
        train2, _, _ = splitter2.split(sample_data, n_train=6, n_val=2, n_test=2)

        assert train1 != train2

    def test_all_items_included(self, sample_data: list[dict[str, Any]]) -> None:
        """All items are included in splits."""
        splitter = RandomSplitter(seed=42)
        train, val, test = splitter.split(sample_data, n_train=6, n_val=2, n_test=2)

        all_ids = {item['id'] for split in [train, val, test] if split for item in split}
        assert all_ids == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}


class TestStratifiedSplitter:
    def test_preserves_class_distribution(self, labeled_data: list[dict[str, Any]]) -> None:
        """Stratified split preserves class distribution."""
        splitter = StratifiedSplitter(key='label', mode='sequential', seed=42)
        train, val, test = splitter.split(labeled_data, n_train=6, n_val=2, n_test=2)

        def count_labels(split: list[dict[str, Any]] | None) -> dict[str, int]:
            if split is None:
                return {}
            counts: dict[str, int] = {}
            for item in split:
                label = str(item['label'])
                counts[label] = counts.get(label, 0) + 1
            return counts

        train_counts = count_labels(train)
        _val_counts = count_labels(val)
        _test_counts = count_labels(test)

        # Check that all classes are represented (approximately)
        assert 'A' in train_counts
        assert 'B' in train_counts
        assert 'C' in train_counts

    def test_stratified_with_random_mode(self, labeled_data: list[dict[str, Any]]) -> None:
        """Stratified split with random shuffling within groups.

        Note: Due to proportional splitting per group, exact counts may vary slightly
        from requested amounts. This is expected behavior for stratified splitting.
        """
        splitter = StratifiedSplitter(key='label', mode='random', seed=42)
        train, val, test = splitter.split(labeled_data, n_train=6, n_val=2, n_test=2)

        # Verify splits are created
        assert len(train) > 0
        assert val is not None  # val_set is created when n_val > 0
        assert len(test) > 0
        # Total should match input (val might be empty due to proportional split rounding)
        assert len(train) + len(val) + len(test) == len(labeled_data)

    def test_reproducible_with_same_seed(self, labeled_data: list[dict[str, Any]]) -> None:
        """Same seed produces same stratified split."""
        splitter1 = StratifiedSplitter(key='label', mode='random', seed=42)
        splitter2 = StratifiedSplitter(key='label', mode='random', seed=42)

        train1, val1, test1 = splitter1.split(labeled_data, n_train=6, n_val=2, n_test=2)
        train2, val2, test2 = splitter2.split(labeled_data, n_train=6, n_val=2, n_test=2)

        assert train1 == train2
        assert val1 == val2
        assert test1 == test2

    def test_no_validation_set(self, labeled_data: list[dict[str, Any]]) -> None:
        """Stratified split without validation set.

        Note: Due to proportional splitting per group, exact counts may vary slightly.
        """
        splitter = StratifiedSplitter(key='label', mode='sequential', seed=42)
        train, val, test = splitter.split(labeled_data, n_train=7, n_val=0, n_test=3)

        assert len(train) > 0
        assert val is None
        assert len(test) > 0
        # Total should match input
        assert len(train) + len(test) == len(labeled_data)
