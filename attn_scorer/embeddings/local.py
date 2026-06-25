"""Local sentence-transformer embedding backend."""

from __future__ import annotations

import numpy as np

from ..config import ScorerConfig
from .base import EmbeddingBackend


class LocalEmbeddingBackend(EmbeddingBackend):
    """Embedding via a local sentence-transformers model."""

    def __init__(self, config: ScorerConfig) -> None:
        self._model_name = config.embedding_model
        self._model = None
        self._cache: dict[str, np.ndarray] = {} if config.cache_embeddings else None
        self._dim: int | None = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed(self, text: str) -> np.ndarray:
        if self._cache is not None and text in self._cache:
            return self._cache[text]
        model = self._load()
        vec = model.encode(text, normalize_embeddings=True)
        vec = np.asarray(vec, dtype=np.float32)
        if self._cache is not None:
            self._cache[text] = vec
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if self._cache is None:
            model = self._load()
            return np.asarray(
                model.encode(texts, normalize_embeddings=True), dtype=np.float32
            )

        uncached_idx = []
        uncached_texts = []
        results = [None] * len(texts)

        for i, t in enumerate(texts):
            if t in self._cache:
                results[i] = self._cache[t]
            else:
                uncached_idx.append(i)
                uncached_texts.append(t)

        if uncached_texts:
            model = self._load()
            vecs = np.asarray(
                model.encode(uncached_texts, normalize_embeddings=True),
                dtype=np.float32,
            )
            for j, idx in enumerate(uncached_idx):
                self._cache[uncached_texts[j]] = vecs[j]
                results[idx] = vecs[j]

        return np.array(results, dtype=np.float32)

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._load()
        return self._dim
