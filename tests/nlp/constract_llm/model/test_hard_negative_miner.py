"""Behavioral tests for hard negative miner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bm25s.tokenization import Tokenized

from nlp.constract_llm.model.embedder.model.base import BaseEmbedder
from nlp.constract_llm.model.hard_negative_miner import HardNegativeMiner
from nlp.constract_llm.model.tokenizer.base import BaseTokenizer

if TYPE_CHECKING:
    from collections.abc import Sequence


class RecordingTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.return_id_calls: list[bool] = []

    def tokenize(self, texts: str | list[str], return_ids: bool = True) -> list[list[str]] | Tokenized:
        self.return_id_calls.append(return_ids)
        if isinstance(texts, str):
            texts = [texts]
        if return_ids:
            ids = [[index for index, _ in enumerate(text)] for text in texts]
            return Tokenized(ids=ids, vocab={'dummy': 0})
        return [[f'{text}-tok'] for text in texts]


class DummyEmbedder(BaseEmbedder):
    def __init__(self, *, requires_token_ids: bool, retrieval_order: Sequence[int]) -> None:
        self._requires_token_ids = requires_token_ids
        self._retrieval_order = list(retrieval_order)
        self.seen_queries: list[list[str] | Tokenized] = []

    @property
    def requires_token_ids(self) -> bool:
        return self._requires_token_ids

    def fit(self, tokenized_corpus: list[list[str]] | Tokenized) -> None:  # pragma: no cover - unused
        _ = tokenized_corpus

    def retrieve(
        self,
        tokenized_query: list[str] | Tokenized,
        corpus: list[str],
        k: int = 1,
    ) -> tuple[list[str], list[float]]:
        self.seen_queries.append(tokenized_query)
        order = self._retrieval_order[:k]
        docs = [corpus[idx] for idx in order]
        scores = [float(k - idx) for idx in range(len(order))]
        return docs, scores

    def save(self, path: str, **kwargs: object) -> None:  # pragma: no cover - unused
        _ = path
        _ = kwargs

    @classmethod
    def load(cls, path: str, **kwargs: object) -> DummyEmbedder:  # pragma: no cover - unused
        _ = path
        _ = kwargs
        return cls(requires_token_ids=False, retrieval_order=[])


def _run_miner(requires_token_ids: bool) -> tuple[dict[int, list[int]], RecordingTokenizer, DummyEmbedder]:
    embedder = DummyEmbedder(requires_token_ids=requires_token_ids, retrieval_order=[1, 0])
    tokenizer = RecordingTokenizer()
    miner = HardNegativeMiner(embedder, tokenizer, num_negatives=1)

    negatives = miner.mine(['query'], [0], ['doc0', 'doc1'])
    return negatives, tokenizer, embedder


def test_miner_obeys_token_id_requirement() -> None:
    negatives, tokenizer, embedder = _run_miner(requires_token_ids=True)

    assert negatives[0] == [1]
    assert tokenizer.return_id_calls == [True]
    assert isinstance(embedder.seen_queries[0], Tokenized)


def test_miner_returns_str_tokens_when_not_required() -> None:
    negatives, tokenizer, embedder = _run_miner(requires_token_ids=False)

    assert negatives[0] == [1]
    assert tokenizer.return_id_calls == [False]
    assert isinstance(embedder.seen_queries[0], list)
