from abc import ABC, abstractmethod

from bm25s.tokenization import Tokenized


class BaseEmbedder(ABC):
    @abstractmethod
    def fit(self, tokenized_corpus: list[list[str]] | Tokenized) -> None: ...

    @abstractmethod
    def retrieve(
        self, tokenized_query: list[str] | Tokenized, corpus: list[str], k: int = 1
    ) -> tuple[list[str], list[float]]: ...

    @abstractmethod
    def save(self, path: str, **kwargs: object) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str, **kwargs: object) -> 'BaseEmbedder': ...
