from typing import Literal

from bm25s.tokenization import Tokenized
from sudachipy import dictionary as sudachi_dict_module
from sudachipy import tokenizer as sudachi_tokenizer
from tqdm.auto import tqdm

from nlp.constract_llm.model.tokenizer.base import BaseTokenizer


class SudachiTokenizer(BaseTokenizer):
    def __init__(
        self,
        sudachi_mode: Literal['A', 'B', 'C'] = 'C',
        sudachi_dict: Literal['small', 'core', 'full'] = 'core',
        stopwords: list[str] | None = None,
        pos_filter: list[str] | None = None,
        show_progress: bool = True,
        leave: bool = False,
    ):
        super().__init__(stopwords, pos_filter, show_progress, leave)
        self.mode = getattr(sudachi_tokenizer.Tokenizer.SplitMode, sudachi_mode)
        self.tokenizer = sudachi_dict_module.Dictionary(dict=sudachi_dict).create()

    def tokenize(self, texts: str | list[str], return_ids: bool = True) -> list[list[str]] | Tokenized:
        if isinstance(texts, str):
            texts = [texts]

        corpus_ids: list[list[int]] = []
        token_to_index: dict[str, int] = {}

        for text in tqdm(
            texts,
            desc='Sudachi Tokenizing',
            disable=not self.show_progress,
            leave=self.leave,
        ):
            morphemes = self.tokenizer.tokenize(text, self.mode)
            tokens: list[str] = []
            for m in morphemes:
                if self.pos_filter is None or any(m.part_of_speech()[0].startswith(pos) for pos in self.pos_filter):
                    token = m.normalized_form()
                    if token not in self.stopwords:
                        tokens.append(token)

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
