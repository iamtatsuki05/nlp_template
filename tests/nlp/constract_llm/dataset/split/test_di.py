"""Minimal DI tests for split strategies."""

from typing import cast

import pytest

from nlp.constract_llm.dataset.split.di import create_split_strategy
from nlp.constract_llm.dataset.split.strategy import (
    RandomSplitter,
    SequentialSplitter,
    SplitMode,
    StratifiedSplitter,
)


@pytest.mark.parametrize(
    ('mode', 'random_seed', 'stratify_key', 'expected'),
    [
        ('sequential', 0, None, SequentialSplitter),
        ('random', 0, None, RandomSplitter),
        ('stratified', 0, 'label', StratifiedSplitter),
    ],
)
def test_create_split_strategy_returns_expected_strategy(
    mode: SplitMode,
    random_seed: int,
    stratify_key: str | None,
    expected: type[object],
) -> None:
    strategy = create_split_strategy(mode=mode, random_seed=random_seed, stratify_key=stratify_key)
    assert isinstance(strategy, expected)


def test_stratified_strategy_requires_stratify_key() -> None:
    with pytest.raises(ValueError, match='stratify_key'):
        create_split_strategy(mode='stratified', random_seed=0, stratify_key=None)


def test_invalid_strategy_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match='Unsupported split mode'):
        create_split_strategy(mode=cast('SplitMode', 'unknown'))
