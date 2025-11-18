"""Tests for model factories dependency injection module."""

from __future__ import annotations

import pytest
from injector import Injector

from nlp.constract_llm.model.di import (
    ModelFactoriesModule,
    create_embedder,
    create_tokenizer,
)
from nlp.constract_llm.model.embedder.model.base import BaseEmbedder
from nlp.constract_llm.model.factories import (
    EmbedderFactory,
    EmbedderParams,
    TokenizerFactory,
    TokenizerParams,
)
from nlp.constract_llm.model.tokenizer.base import BaseTokenizer


class TestModelFactoriesModule:
    """Test ModelFactoriesModule."""

    def test_module_provides_tokenizer_factory(self) -> None:
        """Test that module provides TokenizerFactory."""
        module = ModelFactoriesModule()
        injector = Injector([module])

        factory = injector.get(TokenizerFactory)  # type: ignore[type-abstract]
        assert factory is not None

    def test_module_provides_embedder_factory(self) -> None:
        """Test that module provides EmbedderFactory."""
        module = ModelFactoriesModule()
        injector = Injector([module])

        factory = injector.get(EmbedderFactory)  # type: ignore[type-abstract]
        assert factory is not None

    def test_tokenizer_factory_creates_sudachi_tokenizer(self) -> None:
        """Test that TokenizerFactory can create Sudachi tokenizer."""
        module = ModelFactoriesModule()
        injector = Injector([module])
        factory = injector.get(TokenizerFactory)  # type: ignore[type-abstract]

        tokenizer = factory.build('sudachi')
        assert isinstance(tokenizer, BaseTokenizer)

    def test_tokenizer_factory_creates_mecab_tokenizer(self) -> None:
        """Test that TokenizerFactory can create MeCab tokenizer."""
        module = ModelFactoriesModule()
        injector = Injector([module])
        factory = injector.get(TokenizerFactory)  # type: ignore[type-abstract]

        tokenizer = factory.build('mecab')
        assert isinstance(tokenizer, BaseTokenizer)

    def test_embedder_factory_creates_bm25s_embedder(self) -> None:
        """Test that EmbedderFactory can create BM25S embedder."""
        module = ModelFactoriesModule()
        injector = Injector([module])
        factory = injector.get(EmbedderFactory)  # type: ignore[type-abstract]

        embedder = factory.build('bm25s')
        assert isinstance(embedder, BaseEmbedder)

    def test_embedder_factory_creates_tfidf_embedder(self) -> None:
        """Test that EmbedderFactory can create TF-IDF embedder."""
        module = ModelFactoriesModule()
        injector = Injector([module])
        factory = injector.get(EmbedderFactory)  # type: ignore[type-abstract]

        embedder = factory.build('tfidf')
        assert isinstance(embedder, BaseEmbedder)

    def test_singleton_returns_same_tokenizer_factory_instance(self) -> None:
        """Test that module returns the same TokenizerFactory instance (singleton)."""
        module = ModelFactoriesModule()
        injector = Injector([module])

        factory1 = injector.get(TokenizerFactory)  # type: ignore[type-abstract]
        factory2 = injector.get(TokenizerFactory)  # type: ignore[type-abstract]

        assert factory1 is factory2

    def test_singleton_returns_same_embedder_factory_instance(self) -> None:
        """Test that module returns the same EmbedderFactory instance (singleton)."""
        module = ModelFactoriesModule()
        injector = Injector([module])

        factory1 = injector.get(EmbedderFactory)  # type: ignore[type-abstract]
        factory2 = injector.get(EmbedderFactory)  # type: ignore[type-abstract]

        assert factory1 is factory2


class TestCreateTokenizer:
    """Test create_tokenizer function."""

    def test_creates_sudachi_tokenizer_with_default_params(self) -> None:
        """Test that function creates Sudachi tokenizer with default params."""
        tokenizer = create_tokenizer('sudachi')

        assert isinstance(tokenizer, BaseTokenizer)

    def test_creates_mecab_tokenizer(self) -> None:
        """Test that function creates MeCab tokenizer."""
        tokenizer = create_tokenizer('mecab')

        assert isinstance(tokenizer, BaseTokenizer)

    def test_creates_tokenizer_with_custom_params(self) -> None:
        """Test that function creates tokenizer with custom params."""
        params = TokenizerParams(
            sudachi_mode='A',
            sudachi_dict='core',  # Use 'core' instead of 'small' as it's installed
            stopwords=['の', 'に'],
        )
        tokenizer = create_tokenizer('sudachi', params=params)

        assert isinstance(tokenizer, BaseTokenizer)

    def test_raises_error_for_invalid_tokenizer_type(self) -> None:
        """Test that function raises ValueError for invalid tokenizer type."""
        with pytest.raises(ValueError, match='Unsupported tokenizer type'):
            create_tokenizer('invalid_type')

    def test_uses_provided_factory(self) -> None:
        """Test that function uses provided factory instead of creating new one."""

        class MockFactory:
            def build(self, _token_type: str, _params: TokenizerParams | None = None) -> BaseTokenizer:
                # Return a simple mock tokenizer
                return create_tokenizer('sudachi')

        mock_factory = MockFactory()
        tokenizer = create_tokenizer('test', factory=mock_factory)

        assert isinstance(tokenizer, BaseTokenizer)


class TestCreateEmbedder:
    """Test create_embedder function."""

    def test_creates_bm25s_embedder_with_default_params(self) -> None:
        """Test that function creates BM25S embedder with default params."""
        embedder = create_embedder('bm25s')

        assert isinstance(embedder, BaseEmbedder)

    def test_creates_tfidf_embedder(self) -> None:
        """Test that function creates TF-IDF embedder."""
        embedder = create_embedder('tfidf')

        assert isinstance(embedder, BaseEmbedder)

    def test_creates_gensim_bm25_embedder(self) -> None:
        """Test that function creates Gensim BM25 embedder."""
        embedder = create_embedder('gensim_bm25')

        assert isinstance(embedder, BaseEmbedder)

    def test_creates_embedder_with_custom_params(self) -> None:
        """Test that function creates embedder with custom params."""
        # corpus should be list of strings, not list of lists
        params = EmbedderParams(corpus=['text1', 'text2'])
        embedder = create_embedder('bm25s', params=params)

        assert isinstance(embedder, BaseEmbedder)

    def test_raises_error_for_invalid_embedder_type(self) -> None:
        """Test that function raises ValueError for invalid embedder type."""
        with pytest.raises(ValueError, match='Unsupported embedder type'):
            create_embedder('invalid_type')

    def test_uses_provided_factory(self) -> None:
        """Test that function uses provided factory instead of creating new one."""

        class MockFactory:
            def build(self, _embedder_type: str, _params: EmbedderParams | None = None) -> BaseEmbedder:
                # Return a simple mock embedder
                return create_embedder('bm25s')

        mock_factory = MockFactory()
        embedder = create_embedder('test', factory=mock_factory)

        assert isinstance(embedder, BaseEmbedder)
