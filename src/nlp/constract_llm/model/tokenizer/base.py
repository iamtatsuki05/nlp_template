from abc import ABC, abstractmethod

from bm25s.tokenization import Tokenized


class BaseTokenizer(ABC):
    def __init__(
        self,
        stopwords: list[str] | None = None,
        pos_filter: list[str] | None = None,
        show_progress: bool = True,
        leave: bool = False,
    ) -> None:
        self.stopwords = set(stopwords) if stopwords else set()
        self.pos_filter = pos_filter
        self.show_progress = show_progress
        self.leave = leave

    @abstractmethod
    def tokenize(self, texts: str | list[str], return_ids: bool = True) -> list[list[str]] | Tokenized: ...
