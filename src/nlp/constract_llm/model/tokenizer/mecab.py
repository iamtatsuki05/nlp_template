import MeCab
from bm25s.tokenization import Tokenized
from tqdm.auto import tqdm

from nlp.constract_llm.model.tokenizer.base import BaseTokenizer


class MeCabTokenizer(BaseTokenizer):
    def __init__(
        self,
        stopwords: list[str] | None = None,
        pos_filter: list[str] | None = None,
        show_progress: bool = True,
        leave: bool = False,
    ):
        super().__init__(stopwords, pos_filter, show_progress, leave)
        self.tagger = MeCab.Tagger()

    def tokenize(self, texts: str | list[str], return_ids: bool = True) -> list[list[str]] | Tokenized:
        if isinstance(texts, str):
            texts = [texts]

        corpus_ids: list[list[int]] = []
        token_to_index: dict[str, int] = {}

        for text in tqdm(
            texts,
            desc='MeCab Tokenizing',
            disable=not self.show_progress,
            leave=self.leave,
        ):
            node = self.tagger.parseToNode(text)
            tokens: list[str] = []
            while node:
                surface = node.surface
                if surface:
                    pos = node.feature.split(',')[0]
                    if (self.pos_filter is None or pos in self.pos_filter) and (surface not in self.stopwords):
                        tokens.append(surface)
                node = node.next

            doc_ids: list[int] = []
            for token in tokens:
                if token not in token_to_index:
                    token_to_index[token] = len(token_to_index)
                doc_ids.append(token_to_index[token])

            corpus_ids.append(doc_ids)

        if return_ids:
            return Tokenized(ids=corpus_ids, vocab=token_to_index)
        else:
            rev = {v: k for k, v in token_to_index.items()}
            return [[rev[i] for i in doc] for doc in corpus_ids]
