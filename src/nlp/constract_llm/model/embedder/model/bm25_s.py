import bm25s

from nlp.constract_llm.model.embedder.model.base import BaseEmbedder


class BM25SModel(BaseEmbedder):
    def __init__(self, corpus: list[str] | None = None) -> None:
        self.model = bm25s.BM25(corpus=corpus)

    @property
    def requires_token_ids(self) -> bool:
        return True

    def fit(self, tokenized_corpus: list[list[str]]) -> None:
        self.model.index(tokenized_corpus)

    def retrieve(self, tokenized_query: list[str], corpus: list[str], k: int = 1) -> tuple[list[str], list[float]]:
        results, scores = self.model.retrieve(tokenized_query, corpus=corpus, k=k)
        return results[0].tolist(), scores[0].tolist()

    def save(self, path: str, **_: object) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str, **kwargs: object) -> 'BM25SModel':
        load_corpus = bool(kwargs.get('load_corpus', False))
        loaded = bm25s.BM25.load(path, load_corpus=load_corpus)
        inst = cls()
        inst.model = loaded
        return inst
