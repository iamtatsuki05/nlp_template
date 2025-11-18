"""Behavioral tests for the transformers model class registry."""

from typing import TYPE_CHECKING

import pytest
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
)

from nlp.constract_llm.model.model_factory import ModelType, TransformersModelClassRegistry

if TYPE_CHECKING:
    from transformers.models.auto.auto_factory import _BaseAutoModelClass


def test_get_model_class_returns_expected_class() -> None:
    registry = TransformersModelClassRegistry()

    mapping: list[tuple[ModelType, type[_BaseAutoModelClass | AutoModel]]] = [
        ('seq2seq', AutoModelForSeq2SeqLM),
        ('causal', AutoModelForCausalLM),
        ('masked', AutoModelForMaskedLM),
        ('generic', AutoModel),
    ]

    for model_type, expected in mapping:
        assert registry.get_model_class(model_type) is expected


def test_get_model_class_invalid_type_raises() -> None:
    registry = TransformersModelClassRegistry()

    with pytest.raises(ValueError, match='Invalid model_type'):
        registry.get_model_class('unknown')  # type: ignore[arg-type]


def test_get_supported_types_includes_known_keys() -> None:
    supported = TransformersModelClassRegistry.get_supported_types()

    assert {'seq2seq', 'causal', 'masked', 'generic'} <= set(supported)
