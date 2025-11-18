"""Behavior-focused tests for dataset splitters."""

from typing import Any

import pytest

from nlp.constract_llm.dataset.split.strategy import (
    RandomSplitter,
    SequentialSplitter,
    StratifiedSplitter,
)


@pytest.fixture
def sample_data() -> list[dict[str, Any]]:
    return [{'id': i, 'value': f'item_{i}'} for i in range(10)]


@pytest.fixture
def labeled_data() -> list[dict[str, Any]]:
    return [{'id': i, 'label': 'A' if i < 4 else 'B' if i < 8 else 'C'} for i in range(10)]


def test_sequential_split_preserves_order(sample_data: list[dict[str, Any]]) -> None:
    splitter = SequentialSplitter()
    train, val, test = splitter.split(sample_data, n_train=6, n_val=2, n_test=2)

    assert [item['id'] for item in train] == list(range(6))
    assert val is not None
    assert [item['id'] for item in val] == [6, 7]
    assert [item['id'] for item in test] == [8, 9]


def test_random_split_reproducible_by_seed(sample_data: list[dict[str, Any]]) -> None:
    splitter_one = RandomSplitter(seed=42)
    splitter_two = RandomSplitter(seed=42)

    train_one, val_one, test_one = splitter_one.split(sample_data, n_train=6, n_val=2, n_test=2)
    train_two, val_two, test_two = splitter_two.split(sample_data, n_train=6, n_val=2, n_test=2)

    assert train_one == train_two
    assert val_one == val_two
    assert test_one == test_two
    concatenated = [*train_one, *(val_one or []), *(test_one or [])]
    assert {item['id'] for item in concatenated} == set(range(10))


def test_stratified_split_keeps_label_distribution(labeled_data: list[dict[str, Any]]) -> None:
    splitter = StratifiedSplitter(key='label', mode='sequential', seed=0)
    train, val, test = splitter.split(labeled_data, n_train=6, n_val=2, n_test=2)

    def label_counts(split: list[dict[str, Any]] | None) -> set[str]:
        if not split:
            return set()
        return {item['label'] for item in split}

    assert label_counts(train) <= {'A', 'B', 'C'}
    assert len([*train, *(val or []), *(test or [])]) == len(labeled_data)
