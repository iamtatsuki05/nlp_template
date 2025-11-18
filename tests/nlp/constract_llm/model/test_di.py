"""Behavioral tests for model factories DI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from injector import Injector

from nlp.constract_llm.model.di import ModelFactoriesModule, create_embedder, create_tokenizer
from nlp.constract_llm.model.embedder.model.base import BaseEmbedder
from nlp.constract_llm.model.factories import EmbedderFactory, EmbedderParams, TokenizerFactory
from nlp.constract_llm.model.tokenizer.base import BaseTokenizer

if TYPE_CHECKING:
    from bm25s.tokenization import Tokenized


def test_model_factories_module_produces_singleton_factories() -> None:
    module = ModelFactoriesModule()
    injector = Injector([module])

    tokenizer_factory = injector.get(TokenizerFactory)  # type: ignore[type-abstract]
    assert tokenizer_factory is injector.get(TokenizerFactory)  # type: ignore[type-abstract]

    embedder_factory = injector.get(EmbedderFactory)  # type: ignore[type-abstract]
    assert embedder_factory is injector.get(EmbedderFactory)  # type: ignore[type-abstract]


@pytest.mark.parametrize('tokenizer_type', ['sudachi', 'mecab'])
def test_create_tokenizer_returns_base(tokenizer_type: str) -> None:
    tokenizer = create_tokenizer(tokenizer_type)
    assert isinstance(tokenizer, BaseTokenizer)


def test_create_tokenizer_invalid_type_raises() -> None:
    with pytest.raises(ValueError, match='Unsupported tokenizer type'):
        create_tokenizer('unknown')


def test_create_tokenizer_uses_custom_factory() -> None:
    class SimpleFactory:
        def build(self, *_: object, **__: object) -> BaseTokenizer:
            class SimpleTokenizer(BaseTokenizer):
                def tokenize(self, texts: str | list[str], return_ids: bool = True) -> list[list[str]]:
                    suffix = '-id' if return_ids else ''
                    if isinstance(texts, str):
                        texts = [texts]
                    return [[f'{text}{suffix}'] for text in texts]

            return SimpleTokenizer()

    tokenizer = create_tokenizer('custom', factory=SimpleFactory())
    assert isinstance(tokenizer, BaseTokenizer)


@pytest.mark.parametrize('embedder_type', ['bm25s', 'tfidf', 'gensim_bm25'])
def test_create_embedder_returns_base(embedder_type: str) -> None:
    embedder = create_embedder(embedder_type)
    assert isinstance(embedder, BaseEmbedder)


def test_create_embedder_invalid_type_raises() -> None:
    with pytest.raises(ValueError, match='Unsupported embedder type'):
        create_embedder('unknown')


def test_create_embedder_uses_custom_factory() -> None:
    class SimpleFactory:
        def build(self, *_: object, **__: object) -> BaseEmbedder:
            class SimpleEmbedder(BaseEmbedder):
                @property
                def requires_token_ids(self) -> bool:
                    return False

                def fit(self, tokenized_corpus: list[list[str]] | Tokenized, **kwargs: object) -> None:
                    _ = tokenized_corpus
                    _ = kwargs

                def retrieve(
                    self,
                    tokenized_query: list[str] | Tokenized,
                    corpus: list[str],
                    k: int = 1,
                ) -> tuple[list[str], list[float]]:
                    _ = tokenized_query
                    _ = corpus
                    _ = k
                    return [], []

                def save(self, path: str, **kwargs: object) -> None:
                    _ = path
                    _ = kwargs

                @classmethod
                def load(cls, path: str, **kwargs: object) -> BaseEmbedder:
                    _ = path
                    _ = kwargs
                    return cls()

            return SimpleEmbedder()

    embedder = create_embedder('custom', factory=SimpleFactory())
    assert isinstance(embedder, BaseEmbedder)


def test_create_embedder_accepts_params() -> None:
    params = EmbedderParams(corpus=['text1', 'text2'])
    embedder = create_embedder('bm25s', params=params)
    assert isinstance(embedder, BaseEmbedder)
