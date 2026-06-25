"""Embedding backends for attn_scorer."""

from .base import EmbeddingBackend
from .local import LocalEmbeddingBackend

__all__ = ["EmbeddingBackend", "LocalEmbeddingBackend"]
