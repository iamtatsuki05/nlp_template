"""Tests for model factory module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
)

from nlp.constract_llm.model.model_factory import (
    ModelType,
    TransformersModelClassRegistry,
)

if TYPE_CHECKING:
    from transformers.models.auto.auto_factory import _BaseAutoModelClass


class TestTransformersModelClassRegistry:
    """Test TransformersModelClassRegistry."""

    @pytest.fixture
    def registry(self) -> TransformersModelClassRegistry:
        """Create a registry instance."""
        return TransformersModelClassRegistry()

    @pytest.mark.parametrize(
        ('model_type', 'expected_class'),
        [
            ('seq2seq', AutoModelForSeq2SeqLM),
            ('causal', AutoModelForCausalLM),
            ('masked', AutoModelForMaskedLM),
            ('generic', AutoModel),
        ],
    )
    def test_get_model_class_returns_correct_class(
        self,
        registry: TransformersModelClassRegistry,
        model_type: ModelType,
        expected_class: type[_BaseAutoModelClass],
    ) -> None:
        """Test that get_model_class returns the correct model class."""
        result = registry.get_model_class(model_type)
        assert result is expected_class

    def test_get_model_class_raises_on_invalid_type(
        self,
        registry: TransformersModelClassRegistry,
    ) -> None:
        """Test that get_model_class raises ValueError for invalid model type."""
        with pytest.raises(ValueError, match="Invalid model_type 'invalid'"):
            registry.get_model_class('invalid')  # type: ignore[arg-type]

    def test_get_supported_types_returns_all_types(self) -> None:
        """Test that get_supported_types returns all registered model types."""
        supported_types = TransformersModelClassRegistry.get_supported_types()

        assert 'seq2seq' in supported_types
        assert 'causal' in supported_types
        assert 'masked' in supported_types
        assert 'generic' in supported_types
        assert len(supported_types) == 4

    def test_register_adds_new_model_type(
        self,
        registry: TransformersModelClassRegistry,
    ) -> None:
        """Test that register method adds a new model type to the registry."""
        # Register a custom model type
        TransformersModelClassRegistry.register('custom', AutoModel)  # type: ignore[arg-type]

        result = registry.get_model_class('custom')  # type: ignore[arg-type]
        assert result is AutoModel

        # Note: No cleanup needed as this affects class-level state
        # which is acceptable in test isolation

    def test_register_overwrites_existing_model_type(
        self,
        registry: TransformersModelClassRegistry,
    ) -> None:
        """Test that register can overwrite an existing model type."""
        original_class = registry.get_model_class('generic')
        assert original_class is AutoModel

        # Overwrite with a different class
        TransformersModelClassRegistry.register('generic', AutoModelForCausalLM)

        result = registry.get_model_class('generic')
        assert result is AutoModelForCausalLM

        # Restore original
        TransformersModelClassRegistry.register('generic', AutoModel)
