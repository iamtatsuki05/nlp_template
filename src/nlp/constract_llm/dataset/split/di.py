"""Dependency injection module for dataset splitting strategies.

Provides Injector configuration for selecting and configuring split strategies.
"""

from __future__ import annotations

from injector import Injector, Module, provider, singleton

from nlp.constract_llm.dataset.split.strategy import (
    RandomSplitter,
    SequentialSplitter,
    SplitMode,
    SplitStrategy,
    StratifiedSplitter,
)


class SplitStrategyModule(Module):
    """Injector module for configuring dataset split strategies.

    Args:
        mode: Split mode ('random', 'sequential', or 'stratified')
        random_seed: Random seed for reproducible shuffling
        stratify_key: Field name to stratify by (only used when mode='stratified')

    """

    def __init__(
        self,
        mode: SplitMode = 'random',
        random_seed: int = 42,
        stratify_key: str | None = None,
    ) -> None:
        self._mode = mode
        self._random_seed = random_seed
        self._stratify_key = stratify_key

    @singleton
    @provider
    def sequential_splitter(self) -> SequentialSplitter:
        """Provide a sequential splitter instance."""
        return SequentialSplitter()

    @singleton
    @provider
    def random_splitter(self) -> RandomSplitter:
        """Provide a random splitter instance with configured seed."""
        return RandomSplitter(seed=self._random_seed)

    @singleton
    @provider
    def stratified_splitter(self) -> StratifiedSplitter:
        """Provide a stratified splitter instance with configured parameters."""
        if self._stratify_key is None:
            msg = 'stratify_key must be provided for stratified splitting'
            raise ValueError(msg)
        return StratifiedSplitter(
            key=self._stratify_key,
            mode='random',  # Stratified mode always uses random within groups
            seed=self._random_seed,
        )

    @property
    def mode(self) -> SplitMode:
        """Get the configured split mode."""
        return self._mode


def create_split_strategy(
    mode: SplitMode = 'random',
    random_seed: int = 42,
    stratify_key: str | None = None,
) -> SplitStrategy:
    """Create a split strategy via dependency injection.

    Args:
        mode: Split mode to use
        random_seed: Random seed for reproducibility
        stratify_key: Field to stratify by (required if mode='stratified')

    Returns:
        Configured SplitStrategy instance

    Raises:
        ValueError: If stratify_key is None when mode='stratified'

    Example:
        >>> strategy = create_split_strategy(mode='random', random_seed=42)
        >>> train, val, test = strategy.split(data, n_train=80, n_val=10, n_test=10)

    """
    module = SplitStrategyModule(
        mode=mode,
        random_seed=random_seed,
        stratify_key=stratify_key,
    )
    injector = Injector([module])

    match module.mode:
        case 'sequential':
            return injector.get(SequentialSplitter)
        case 'random':
            return injector.get(RandomSplitter)
        case 'stratified':
            return injector.get(StratifiedSplitter)
        case _:
            msg = f'Unsupported split mode: {module.mode}'
            raise ValueError(msg)
