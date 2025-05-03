from gensim import corpora, models, similarities

from nlp.constract_llm.model.embedder.model.base import BaseEmbedder


class GensimTfidfModel(BaseEmbedder):
    def __init__(self):
        self.dictionary: corpora.Dictionary | None = None
        self.tfidf: models.TfidfModel | None = None
        self.index: similarities.Similarity | None = None

    def fit(self, tokenized_corpus: list[list[str]]) -> None:
        self.dictionary = corpora.Dictionary(tokenized_corpus)
        corpus_bow = [self.dictionary.doc2bow(doc) for doc in tokenized_corpus]
        self.tfidf = models.TfidfModel(corpus_bow)
        self.index = similarities.MatrixSimilarity(self.tfidf[corpus_bow], num_features=len(self.dictionary))

    def retrieve(self, tokenized_query: list[str], corpus: list[str], k: int = 1) -> tuple[list[str], list[float]]:
        if not (self.dictionary and self.tfidf and self.index):
            raise ValueError('Model not fitted yet.')
        query_bow = self.dictionary.doc2bow(tokenized_query)
        query_tf = self.tfidf[query_bow]
        sims = list(enumerate(self.index[query_tf]))
        ranked = sorted(sims, key=lambda x: x[1], reverse=True)[:k]
        docs = [corpus[i] for i, _ in ranked]
        scs = [float(score) for _, score in ranked]
        return docs, scs

    def save(self, path: str) -> None:
        # 保存: dictionary, tfidf model, index
        self.dictionary.save(f'{path}.dict')
        self.tfidf.save(f'{path}.tfidf')
        self.index.save(f'{path}.index')

    @classmethod
    def load(cls, path: str) -> 'GensimTfidfModel':
        inst = cls()
        inst.dictionary = corpora.Dictionary.load(f'{path}.dict')
        inst.tfidf = models.TfidfModel.load(f'{path}.tfidf')
        inst.index = similarities.MatrixSimilarity.load(f'{path}.index')
        return inst
