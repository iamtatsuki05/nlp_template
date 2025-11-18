from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

from nlp.constract_llm.model.embedder.model.bm25 import GensimBM25Model
from nlp.constract_llm.model.embedder.model.bm25_s import BM25SModel
from nlp.constract_llm.model.embedder.model.tfidf import GensimTfidfModel
from nlp.constract_llm.model.tokenizer.mecab import MeCabTokenizer
from nlp.constract_llm.model.tokenizer.sudachi import SudachiTokenizer

if TYPE_CHECKING:
    from nlp.constract_llm.model.embedder.model.base import BaseEmbedder
    from nlp.constract_llm.model.tokenizer.base import BaseTokenizer, BaseTokenizerKwargs


from pydantic import BaseModel


class TokenizerParams(BaseModel):
    sudachi_mode: Literal['A', 'B', 'C'] = 'C'
    sudachi_dict: Literal['small', 'core', 'full'] = 'core'
    stopwords: list[str] | None = None
    pos_filter: list[str] | None = None


class TokenizerFactory(Protocol):
    def build(self, token_type: str, params: TokenizerParams | None = None) -> BaseTokenizer: ...


type TokenizerBuilder = TokenizerFactory


class EmbedderParams(BaseModel):
    embedder_path: str | None = None
    load_corpus: bool | None = None
    corpus: list[str] | None = None


class EmbedderFactory(Protocol):
    def build(self, embedder_type: str, params: EmbedderParams | None = None) -> BaseEmbedder: ...


type EmbedderBuilder = EmbedderFactory


class DefaultTokenizerFactory:
    def build(self, token_type: str, params: TokenizerParams | None = None) -> BaseTokenizer:
        params = params or TokenizerParams()
        optional_kwargs: BaseTokenizerKwargs = {}
        if params.stopwords is not None:
            optional_kwargs['stopwords'] = params.stopwords
        if params.pos_filter is not None:
            optional_kwargs['pos_filter'] = params.pos_filter
        match token_type:
            case 'sudachi':
                return SudachiTokenizer(
                    sudachi_mode=params.sudachi_mode,
                    sudachi_dict=params.sudachi_dict,
                    **optional_kwargs,
                )
            case 'mecab':
                return MeCabTokenizer(**optional_kwargs)
        msg = f'Unsupported tokenizer type: {token_type}'
        raise ValueError(msg)


class DefaultEmbedderFactory:
    def build(self, embedder_type: str, params: EmbedderParams | None = None) -> BaseEmbedder:
        params = params or EmbedderParams()
        match embedder_type:
            case 'bm25s':
                if params.embedder_path:
                    return BM25SModel.load(str(params.embedder_path), load_corpus=params.load_corpus or True)
                return BM25SModel(corpus=params.corpus)
            case 'gensim_bm25':
                if params.embedder_path:
                    return GensimBM25Model.load(str(params.embedder_path))
                return GensimBM25Model()
            case 'tfidf':
                if params.embedder_path:
                    return GensimTfidfModel.load(str(params.embedder_path))
                return GensimTfidfModel()
        msg = f'Unsupported embedder type: {embedder_type}'
        raise ValueError(msg)
