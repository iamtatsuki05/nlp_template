"""Dependency injection module for model factories."""

from __future__ import annotations

from typing import TYPE_CHECKING

from injector import Injector, Module, provider, singleton

from nlp.constract_llm.model.factories import (
    DefaultEmbedderFactory,
    DefaultTokenizerFactory,
    EmbedderFactory,
    EmbedderParams,
    TokenizerFactory,
    TokenizerParams,
)

if TYPE_CHECKING:
    from nlp.constract_llm.model.embedder.model.base import BaseEmbedder
    from nlp.constract_llm.model.tokenizer.base import BaseTokenizer


class ModelFactoriesModule(Module):
    """Injector module for model factories.

    This module provides a flexible way to configure and inject model factories
    through dependency injection, making testing easier by allowing mock factories.

    Examples:
        >>> module = ModelFactoriesModule()
        >>> injector = Injector([module])
        >>> tokenizer_factory = injector.get(TokenizerFactory)
        >>> tokenizer = tokenizer_factory.build('sudachi')

    """

    @singleton
    @provider
    def tokenizer_factory(self) -> TokenizerFactory:
        """Provide tokenizer factory instance.

        Returns:
            TokenizerFactory configured to create tokenizers.

        """
        return DefaultTokenizerFactory()

    @singleton
    @provider
    def embedder_factory(self) -> EmbedderFactory:
        """Provide embedder factory instance.

        Returns:
            EmbedderFactory configured to create embedders.

        """
        return DefaultEmbedderFactory()


def create_tokenizer(
    token_type: str,
    params: TokenizerParams | None = None,
    *,
    factory: TokenizerFactory | None = None,
) -> BaseTokenizer:
    """Create tokenizer via dependency injection.

    This function provides a convenient way to create a tokenizer instance
    using dependency injection, which makes testing easier by allowing mock
    factories to be injected.

    Args:
        token_type: Type of tokenizer to create ('sudachi' or 'mecab').
        params: Optional parameters for tokenizer configuration.
        factory: Optional pre-configured factory. If None, creates one via DI.

    Returns:
        BaseTokenizer instance configured with specified parameters.

    Examples:
        >>> tokenizer = create_tokenizer('sudachi')
        >>> tokens = tokenizer.tokenize(['Hello world'])

    """
    if factory is None:
        injector = Injector([ModelFactoriesModule()])
        # Use concrete class for injector.get() to satisfy mypy
        factory = injector.get(DefaultTokenizerFactory)
    return factory.build(token_type, params)


def create_embedder(
    embedder_type: str,
    params: EmbedderParams | None = None,
    *,
    factory: EmbedderFactory | None = None,
) -> BaseEmbedder:
    """Create embedder via dependency injection.

    This function provides a convenient way to create an embedder instance
    using dependency injection, which makes testing easier by allowing mock
    factories to be injected.

    Args:
        embedder_type: Type of embedder to create ('bm25s', 'gensim_bm25', 'tfidf').
        params: Optional parameters for embedder configuration.
        factory: Optional pre-configured factory. If None, creates one via DI.

    Returns:
        BaseEmbedder instance configured with specified parameters.

    Examples:
        >>> embedder = create_embedder('bm25s')
        >>> embedder.fit([['token1', 'token2']])

    """
    if factory is None:
        injector = Injector([ModelFactoriesModule()])
        # Use concrete class for injector.get() to satisfy mypy
        factory = injector.get(DefaultEmbedderFactory)
    return factory.build(embedder_type, params)
