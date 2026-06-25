"""OpenAI embedding backend (supports text-embedding-3-small/large, ada-002)."""

from __future__ import annotations

import numpy as np

from ..config import ScorerConfig
from .base import EmbeddingBackend


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """Embedding via OpenAI API."""

    def __init__(self, config: ScorerConfig) -> None:
        self._api_key = config.api_key
        self._model = config.api_model or "text-embedding-3-small"
        self._client = None
        self._dim: int | None = None
        self._cache: dict[str, np.ndarray] = {} if config.cache_embeddings else None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package required for OpenAI embeddings. "
                    "Install with: pip install openai"
                )
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _call_api(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        response = client.embeddings.create(input=texts, model=self._model)
        vectors = [np.asarray(d.embedding, dtype=np.float32) for d in response.data]
        vecs = np.array(vectors, dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-10)
        if self._dim is None:
            self._dim = vecs.shape[1]
        return vecs

    def embed(self, text: str) -> np.ndarray:
        if self._cache is not None and text in self._cache:
            return self._cache[text]
        vec = self._call_api([text])[0]
        if self._cache is not None:
            self._cache[text] = vec
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not self._cache:
            return self._call_api(texts)

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
            vecs = self._call_api(uncached_texts)
            for j, idx in enumerate(uncached_idx):
                self._cache[uncached_texts[j]] = vecs[j]
                results[idx] = vecs[j]

        return np.array(results, dtype=np.float32)

    @property
    def dimension(self) -> int:
        if self._dim is None:
            # Trigger a call to determine dimension
            self.embed("test")
        return self._dim
