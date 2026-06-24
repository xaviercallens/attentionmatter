"""Embedding service with caching and cosine similarity."""

from __future__ import annotations

import numpy as np

from .config import Config


class EmbeddingService:
    """Computes and caches text embeddings using a local sentence-transformer model."""

    def __init__(self, cfg: Config) -> None:
        self._model_name = cfg.embedding_model
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Return a unit-normalized embedding vector for the given text."""
        if text in self._cache:
            return self._cache[text]
        model = self._load_model()
        vec = model.encode(text, normalize_embeddings=True)
        vec = np.asarray(vec, dtype=np.float32)
        self._cache[text] = vec
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Batch-embed texts, using cache where available."""
        uncached_indices = []
        uncached_texts = []
        results = [None] * len(texts)

        for i, t in enumerate(texts):
            if t in self._cache:
                results[i] = self._cache[t]
            else:
                uncached_indices.append(i)
                uncached_texts.append(t)

        if uncached_texts:
            model = self._load_model()
            vecs = model.encode(uncached_texts, normalize_embeddings=True)
            vecs = np.asarray(vecs, dtype=np.float32)
            for j, idx in enumerate(uncached_indices):
                self._cache[uncached_texts[j]] = vecs[j]
                results[idx] = vecs[j]

        return np.array(results, dtype=np.float32)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two unit-normalized vectors (dot product)."""
        return float(np.dot(a, b))


class DummyEmbeddingService(EmbeddingService):
    """
    Hash-based pseudo-embedding for offline/CI testing.
    Produces deterministic 384-dim vectors from text hashes.
    Similar texts get somewhat similar vectors via keyword overlap.
    """

    def __init__(self, cfg: Config | None = None, dim: int = 384) -> None:
        self._model_name = "dummy"
        self._model = None
        self._cache: dict[str, np.ndarray] = {}
        self._dim = dim

    def _load_model(self):
        return None  # no model needed

    def embed(self, text: str) -> np.ndarray:
        """Generate a deterministic pseudo-embedding from text."""
        if text in self._cache:
            return self._cache[text]
        vec = self._text_to_vector(text)
        self._cache[text] = vec
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts], dtype=np.float32)

    def _text_to_vector(self, text: str) -> np.ndarray:
        """
        Create a pseudo-embedding that preserves some semantic similarity:
        - Uses word-level hashing so texts sharing words get similar vectors.
        - Normalizes to unit length.
        """
        import hashlib

        vec = np.zeros(self._dim, dtype=np.float32)
        words = text.lower().split()
        for word in words:
            # Hash each word to a position and magnitude
            h = hashlib.md5(word.encode()).hexdigest()
            for i in range(0, len(h) - 3, 4):
                idx = int(h[i:i+4], 16) % self._dim
                val = (int(h[i:i+2], 16) - 128) / 128.0
                vec[idx] += val

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            # Fallback: use full text hash for a non-zero vector
            h = hashlib.sha256(text.encode()).digest()
            vec = np.frombuffer(h * (self._dim // 32 + 1), dtype=np.uint8)[:self._dim].astype(np.float32)
            vec = (vec - 128) / 128.0
            vec = vec / np.linalg.norm(vec)

        return vec
