import pickle

from gensim.summarization.bm25 import BM25 as GensimBM25

from nlp.constract_llm.model.embedder.model.base import BaseEmbedder


class GensimBM25Model(BaseEmbedder):
    def __init__(self):
        self.model: GensimBM25 | None = None

    def fit(self, tokenized_corpus: list[list[str]]) -> None:
        self.model = GensimBM25(tokenized_corpus)

    def retrieve(self, tokenized_query: list[str], corpus: list[str], k: int = 1) -> tuple[list[str], list[float]]:
        if self.model is None:
            raise ValueError('Model not fitted yet.')
        scores = self.model.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        docs = [corpus[i] for i, _ in ranked]
        scs = [float(score) for _, score in ranked]
        return docs, scs

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: str) -> 'GensimBM25Model':
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        inst = cls()
        inst.model = loaded
        return inst
