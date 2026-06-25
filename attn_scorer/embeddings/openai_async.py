"""Native async OpenAI embedding backend using httpx."""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from ..config import ScorerConfig
from .base import EmbeddingBackend

logger = logging.getLogger(__name__)


class AsyncOpenAIEmbeddingBackend:
    """
    Native async embedding via OpenAI API using httpx.AsyncClient.
    No thread pool — true async IO for maximum concurrency.
    """

    def __init__(self, config: ScorerConfig, max_retries: int = 3) -> None:
        self._api_key = config.api_key
        self._model = config.api_model or "text-embedding-3-small"
        self._max_retries = max_retries
        self._client = None
        self._dim: int | None = None
        self._cache: dict[str, np.ndarray] = {} if config.cache_embeddings else None

    async def _get_client(self):
        if self._client is None:
            try:
                import httpx
            except ImportError:
                raise ImportError(
                    "httpx required for async OpenAI embeddings. "
                    "Install with: pip install httpx"
                )
            self._client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._client

    async def _call_api(self, texts: list[str]) -> np.ndarray:
        client = await self._get_client()
        payload = {"input": texts, "model": self._model}

        for attempt in range(self._max_retries):
            response = await client.post("/embeddings", json=payload)

            if response.status_code == 200:
                data = response.json()
                vectors = [d["embedding"] for d in data["data"]]
                vecs = np.array(vectors, dtype=np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                vecs = vecs / np.maximum(norms, 1e-10)
                if self._dim is None:
                    self._dim = vecs.shape[1]
                return vecs

            elif response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 2 ** attempt))
                logger.warning("Rate limited. Retrying in %.1fs", retry_after)
                await asyncio.sleep(retry_after)
            else:
                response.raise_for_status()

        raise RuntimeError(f"OpenAI API failed after {self._max_retries} retries")

    async def embed(self, text: str) -> np.ndarray:
        if self._cache is not None and text in self._cache:
            return self._cache[text]
        vec = (await self._call_api([text]))[0]
        if self._cache is not None:
            self._cache[text] = vec
        return vec

    async def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not self._cache:
            return await self._call_api(texts)

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
            vecs = await self._call_api(uncached_texts)
            for j, idx in enumerate(uncached_idx):
                self._cache[uncached_texts[j]] = vecs[j]
                results[idx] = vecs[j]

        return np.array(results, dtype=np.float32)

    @property
    def dimension(self) -> int:
        if self._dim is None:
            raise RuntimeError("Dimension unknown until first embed call")
        return self._dim

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
