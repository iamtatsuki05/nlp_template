"""Model factory with registry pattern for transformers models."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
)

if TYPE_CHECKING:
    from transformers.models.auto.auto_factory import _BaseAutoModelClass

ModelType = Literal['seq2seq', 'causal', 'masked', 'generic']


class ModelClassProvider(Protocol):
    """Protocol for providing model classes based on type."""

    def get_model_class(self, model_type: ModelType) -> type[_BaseAutoModelClass]:
        """Get model class for the specified type.

        Args:
            model_type: Type of model to retrieve.

        Returns:
            Model class corresponding to the specified type.

        Raises:
            ValueError: If model_type is not supported.

        """
        ...


class TransformersModelClassRegistry:
    """Registry-based model class provider for transformers models.

    This class implements the registry pattern to map model types to their
    corresponding AutoModel classes, eliminating the need for match/case statements.

    Examples:
        >>> registry = TransformersModelClassRegistry()
        >>> model_class = registry.get_model_class('causal')
        >>> model = model_class.from_config(config)

    """

    _registry: ClassVar[dict[ModelType, type[_BaseAutoModelClass]]] = {
        'seq2seq': AutoModelForSeq2SeqLM,
        'causal': AutoModelForCausalLM,
        'masked': AutoModelForMaskedLM,
        'generic': AutoModel,
    }

    def get_model_class(self, model_type: ModelType) -> type[_BaseAutoModelClass]:
        """Get model class from registry.

        Args:
            model_type: Type of model to retrieve.

        Returns:
            Model class corresponding to the specified type.

        Raises:
            ValueError: If model_type is not supported.

        """
        model_class = self._registry.get(model_type)
        if model_class is None:
            supported = ', '.join(self._registry.keys())
            msg = f"Invalid model_type '{model_type}'. Supported: {supported}"
            raise ValueError(msg)
        return model_class

    @classmethod
    def register(cls, model_type: ModelType, model_class: type[_BaseAutoModelClass]) -> None:
        """Register a new model type.

        This method allows for extensibility by enabling registration of custom model types.

        Args:
            model_type: Type identifier for the model.
            model_class: Model class to register.

        Examples:
            >>> TransformersModelClassRegistry.register('custom', CustomAutoModel)

        """
        cls._registry[model_type] = model_class

    @classmethod
    def get_supported_types(cls) -> tuple[ModelType, ...]:
        """Get all supported model types.

        Returns:
            Tuple of all supported model type identifiers.

        """
        return tuple(cls._registry.keys())
