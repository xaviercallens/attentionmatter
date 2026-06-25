"""Multi-modal candidate support for diverse content types."""

from .types import ModalityType, MultiModalCandidate
from .scorer import MultiModalScorer
from .extractors import (
    CodeExtractor,
    TableExtractor,
    ImageExtractor,
    StructuredDataExtractor,
)

__all__ = [
    "ModalityType",
    "MultiModalCandidate",
    "MultiModalScorer",
    "CodeExtractor",
    "TableExtractor",
    "ImageExtractor",
    "StructuredDataExtractor",
]
