"""Binary relevance classifier for context scoring."""

from __future__ import annotations

import numpy as np

from ..config import ScorerConfig
from ..models import Candidate


class RelevanceClassifier:
    """
    Feature-based binary classifier: P(relevant | query, candidate).
    Uses logistic regression on hand-crafted features.
    """

    def __init__(self, config: ScorerConfig | None = None, model_path: str | None = None):
        self._config = config or ScorerConfig()
        self._weights: np.ndarray | None = None
        self._bias: float = 0.0
        if model_path:
            self.load(model_path)
        else:
            self._init_default_weights()

    def _init_default_weights(self) -> None:
        # [cosine, decay, is_memory, is_fact, is_chit_chat,
        #  jaccard, keyword_present, text_length_norm]
        self._weights = np.array([
            0.45, 0.15, 0.20, 0.10, -0.10, 0.25, 0.15, 0.02
        ], dtype=np.float32)
        self._bias = -0.15

    def _extract_features(self, query: str, candidate: Candidate, cosine_sim: float) -> np.ndarray:
        decay = self._config.decay_factor ** candidate.age
        is_memory = 1.0 if candidate.ctype == "memory" else 0.0
        is_fact = 1.0 if candidate.ctype == "fact" else 0.0
        is_chit_chat = 1.0 if candidate.ctype == "chit_chat" else 0.0
        q_words = set(query.lower().split())
        c_words = set(candidate.text.lower().split())
        intersection = len(q_words & c_words)
        union = len(q_words | c_words)
        jaccard = intersection / union if union > 0 else 0.0
        keyword_present = 1.0 if intersection > 0 else 0.0
        text_length_norm = min(1.0, len(candidate.text.split()) / 50.0)
        return np.array([
            cosine_sim, decay, is_memory, is_fact, is_chit_chat,
            jaccard, keyword_present, text_length_norm,
        ], dtype=np.float32)

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        exp_x = np.exp(x)
        return float(exp_x / (1.0 + exp_x))

    def predict(self, query: str, candidate: Candidate, cosine_sim: float) -> float:
        features = self._extract_features(query, candidate, cosine_sim)
        logit = float(np.dot(self._weights, features) + self._bias)
        return self._sigmoid(logit)

    def predict_batch(self, query: str, candidates: list[Candidate], cosine_sims: list[float]) -> list[float]:
        features = np.array([
            self._extract_features(query, c, cos)
            for c, cos in zip(candidates, cosine_sims)
        ], dtype=np.float32)
        logits = features @ self._weights + self._bias
        return [self._sigmoid(float(l)) for l in logits]

    def train_feature_model(self, queries, candidates, cosine_sims, labels, lr=0.01, epochs=100):
        n = len(queries)
        features = np.array([
            self._extract_features(q, c, cos)
            for q, c, cos in zip(queries, candidates, cosine_sims)
        ], dtype=np.float32)
        y = np.array(labels, dtype=np.float32)
        self._weights = np.zeros(features.shape[1], dtype=np.float32)
        self._bias = 0.0
        for _ in range(epochs):
            logits = features @ self._weights + self._bias
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
            error = probs - y
            self._weights -= lr * (features.T @ error / n)
            self._bias -= lr * np.mean(error)
        preds = (features @ self._weights + self._bias) > 0
        return {"accuracy": float(np.mean(preds == y))}

    def save(self, path: str) -> None:
        import json
        with open(path, "w") as f:
            json.dump({"weights": self._weights.tolist(), "bias": self._bias}, f)

    def load(self, path: str) -> None:
        import json
        with open(path) as f:
            data = json.load(f)
        self._weights = np.array(data["weights"], dtype=np.float32)
        self._bias = data.get("bias", 0.0)
