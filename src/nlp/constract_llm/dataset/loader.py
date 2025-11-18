from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from datasets import load_dataset
from injector import Injector, Module, provider, singleton

from nlp.common.utils.file.io import load_file

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

JsonRecord = dict[str, Any]


class DatasetLoadResult:
    """Container for dataset splits and their origin information."""

    def __init__(
        self,
        splits: Mapping[str, list[JsonRecord]],
        *,
        source: str,
        source_kind: Literal['local', 'remote'],
    ) -> None:
        if not splits:
            msg = 'Dataset must contain at least one split.'
            raise ValueError(msg)
        # Preserve insertion order for deterministic split selection
        self._splits = OrderedDict((name, list(records)) for name, records in splits.items())
        self._source = source
        self._source_kind = source_kind

    @property
    def source(self) -> str:
        return self._source

    @property
    def is_local(self) -> bool:
        return self._source_kind == 'local'

    @property
    def split_names(self) -> tuple[str, ...]:
        return tuple(self._splits.keys())

    def has_split(self, name: str) -> bool:
        return name in self._splits

    def iter_splits(self) -> Iterable[tuple[str, list[JsonRecord]]]:
        return self._splits.items()

    def pick_split(self, preferred: str | None = None) -> tuple[str, list[JsonRecord]]:
        if preferred and preferred in self._splits:
            return preferred, self._splits[preferred]
        first_name, records = next(iter(self._splits.items()))
        return first_name, records


class DatasetLoader(ABC):
    @abstractmethod
    def load(self) -> DatasetLoadResult: ...


class LocalJsonDatasetLoader(DatasetLoader):
    def __init__(self, path: Path, *, split_name: str) -> None:
        self.path = path
        self.split_name = split_name

    def load(self) -> DatasetLoadResult:
        data = load_file(self.path)
        if not isinstance(data, list):
            raise TypeError(f'Local dataset must be a list, got {type(data).__name__}')
        normalized = [dict(item) for item in data]
        logger.info('Loaded local JSON: %s records from %s', len(normalized), self.path)
        return DatasetLoadResult({self.split_name: normalized}, source=str(self.path), source_kind='local')


class HuggingFaceDatasetLoader(DatasetLoader):
    def __init__(self, dataset_name: str, *, dataset_config: str | None = None) -> None:
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config

    def load(self) -> DatasetLoadResult:
        dataset_dict = (
            load_dataset(self.dataset_name, self.dataset_config)
            if self.dataset_config is not None
            else load_dataset(self.dataset_name)
        )
        splits: dict[str, list[JsonRecord]] = {}
        for split_name in dataset_dict:
            splits[split_name] = [dict(example) for example in dataset_dict[split_name]]
        total_examples = sum(len(records) for records in splits.values())
        logger.info(
            "Loaded Hugging Face dataset '%s' (splits=%s, total=%s)",
            self.dataset_name,
            ','.join(splits),
            total_examples,
        )
        return DatasetLoadResult(splits, source=self.dataset_name, source_kind='remote')


class DatasetLoaderModule(Module):
    def __init__(
        self,
        source: str | Path,
        *,
        dataset_config: str | None = None,
        local_split_name: str = 'default',
        mode: Literal['auto', 'local', 'remote'] = 'auto',
    ) -> None:
        self._source = Path(source)
        self._dataset_config = dataset_config
        self._local_split_name = local_split_name
        self._mode = mode
        self._use_local = self._decide_source_mode()

    def _decide_source_mode(self) -> bool:
        if self._mode == 'local':
            return True
        if self._mode == 'remote':
            return False
        return self._source.exists()

    @property
    def use_local_loader(self) -> bool:
        return self._use_local

    @singleton
    @provider
    def local_loader(self) -> LocalJsonDatasetLoader:
        return LocalJsonDatasetLoader(self._source, split_name=self._local_split_name)

    @singleton
    @provider
    def remote_loader(self) -> HuggingFaceDatasetLoader:
        return HuggingFaceDatasetLoader(str(self._source), dataset_config=self._dataset_config)


def load_dataset_resource(
    source: str | Path,
    *,
    dataset_config: str | None = None,
    local_split_name: str = 'default',
    allow_remote_fallback: bool = False,
) -> DatasetLoadResult:
    """Resolve and load dataset records via Injector-managed loader."""
    module = DatasetLoaderModule(
        source,
        dataset_config=dataset_config,
        local_split_name=local_split_name,
    )
    injector = Injector([module])
    loader: DatasetLoader
    if module.use_local_loader:
        loader = injector.get(LocalJsonDatasetLoader)
    else:
        loader = injector.get(HuggingFaceDatasetLoader)
    try:
        return loader.load()
    except Exception:
        if allow_remote_fallback and Path(source).exists():
            logger.info('Local dataset load failed for %s; trying Hugging Face loader.', source)
            remote_module = DatasetLoaderModule(
                source,
                dataset_config=dataset_config,
                local_split_name=local_split_name,
                mode='remote',
            )
            remote_injector = Injector([remote_module])
            remote_loader = remote_injector.get(HuggingFaceDatasetLoader)
            return remote_loader.load()
        raise


def iter_dataset_records(
    source: str | Path,
    *,
    dataset_config: str | None = None,
    split: str = 'train',
    streaming: bool = False,
) -> Iterator[JsonRecord]:
    """Yield dataset records lazily, with optional HF streaming support."""
    path = Path(source)
    if path.exists():
        data = load_file(path)
        if not isinstance(data, list):
            raise TypeError(f'Local dataset {path} must be a list of records.')
        for item in data:
            yield dict(item)
        return

    if dataset_config is not None:
        dataset = load_dataset(str(source), dataset_config, split=split, streaming=streaming)
    else:
        dataset = load_dataset(str(source), split=split, streaming=streaming)

    for example in dataset:
        yield dict(example)
