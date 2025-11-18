"""Tests for dependency injection of split strategies."""

import pytest
from injector import Injector

from nlp.constract_llm.dataset.split.di import SplitStrategyModule, create_split_strategy
from nlp.constract_llm.dataset.split.strategy import (
    RandomSplitter,
    SequentialSplitter,
    SplitMode,
    StratifiedSplitter,
)


class TestSplitStrategyModule:
    def test_mode_property(self) -> None:
        """Module exposes configured mode."""
        mode: SplitMode = 'random'
        module = SplitStrategyModule(mode=mode)
        assert module.mode == 'random'

    def test_stratified_without_key_raises_error(self) -> None:
        """Stratified mode requires stratify_key."""
        module = SplitStrategyModule(mode='stratified', stratify_key=None)
        injector = Injector([module])

        with pytest.raises(ValueError, match='stratify_key must be provided'):
            injector.get(StratifiedSplitter)


class TestCreateSplitStrategy:
    def test_creates_sequential_strategy(self) -> None:
        """Factory creates sequential splitter."""
        strategy = create_split_strategy(mode='sequential')
        assert isinstance(strategy, SequentialSplitter)

    def test_creates_random_strategy(self) -> None:
        """Factory creates random splitter with seed."""
        strategy = create_split_strategy(mode='random', random_seed=42)
        assert isinstance(strategy, RandomSplitter)
        assert strategy.seed == 42

    def test_creates_stratified_strategy(self) -> None:
        """Factory creates stratified splitter with key."""
        strategy = create_split_strategy(mode='stratified', stratify_key='label', random_seed=123)
        assert isinstance(strategy, StratifiedSplitter)
        assert strategy.key == 'label'
        assert strategy.seed == 123

    def test_stratified_without_key_raises_error(self) -> None:
        """Stratified mode without key raises ValueError."""
        with pytest.raises(ValueError, match='stratify_key must be provided'):
            create_split_strategy(mode='stratified', stratify_key=None)

    def test_different_seeds_create_different_instances(self) -> None:
        """Different seeds create different splitter instances."""
        strategy1 = create_split_strategy(mode='random', random_seed=42)
        strategy2 = create_split_strategy(mode='random', random_seed=123)

        assert isinstance(strategy1, RandomSplitter)
        assert isinstance(strategy2, RandomSplitter)
        assert strategy1.seed != strategy2.seed

    def test_singleton_behavior_within_injector(self) -> None:
        """Same module returns singleton instance."""
        module = SplitStrategyModule(mode='random', random_seed=42)
        injector = Injector([module])

        strategy1 = injector.get(RandomSplitter)
        strategy2 = injector.get(RandomSplitter)

        assert strategy1 is strategy2
