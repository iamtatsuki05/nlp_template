import pickle
from pathlib import Path

from gensim.models import OkapiBM25Model

from nlp.constract_llm.model.embedder.model.base import BaseEmbedder


class GensimBM25Model(BaseEmbedder):
    def __init__(self) -> None:
        self.model: OkapiBM25Model | None = None

    def fit(self, tokenized_corpus: list[list[str]]) -> None:
        self.model = OkapiBM25Model(tokenized_corpus)

    def retrieve(self, tokenized_query: list[str], corpus: list[str], k: int = 1) -> tuple[list[str], list[float]]:
        if self.model is None:
            raise ValueError('Model not fitted yet.')
        scores = self.model.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        docs = [corpus[i] for i, _ in ranked]
        scs = [float(score) for _, score in ranked]
        return docs, scs

    def save(self, path: str, **_: object) -> None:
        if self.model is None:
            msg = 'Model was not fitted. Train before saving.'
            raise ValueError(msg)
        target = Path(path)
        with target.open('wb') as file_obj:
            pickle.dump(self.model, file_obj)

    @classmethod
    def load(cls, path: str, **_: object) -> 'GensimBM25Model':
        source = Path(path)
        with source.open('rb') as file_obj:
            loaded = pickle.load(file_obj)  # noqa: S301 - loading trusted model artifacts only
        inst = cls()
        inst.model = loaded
        return inst
