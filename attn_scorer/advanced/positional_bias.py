"""Trainable positional biases inspired by ALiBi."""

from __future__ import annotations

import json

import numpy as np


class LearnablePositionalBias:
    """
    Learnable position-dependent bias added to relevance scores.
    bias(age) = slope * age + offset + correction[bucket(age)]
    Buckets use logarithmic spacing for efficient age coverage.
    """

    def __init__(self, num_buckets: int = 12, initial_slope: float = -0.005):
        self._num_buckets = num_buckets
        self._slope = initial_slope
        self._offset = 0.0
        self._corrections = np.zeros(num_buckets, dtype=np.float32)
        self._boundaries = [0] + [int(2 ** (i - 1)) for i in range(1, num_buckets)]

    def _get_bucket(self, age: int) -> int:
        for i in range(len(self._boundaries) - 1, -1, -1):
            if age >= self._boundaries[i]:
                return min(i, self._num_buckets - 1)
        return 0

    def bias(self, age: int) -> float:
        linear = self._slope * age + self._offset
        return float(linear + self._corrections[self._get_bucket(age)])

    def bias_batch(self, ages: list[int]) -> np.ndarray:
        linear = np.array(ages, dtype=np.float32) * self._slope + self._offset
        buckets = np.array([self._get_bucket(a) for a in ages])
        return linear + self._corrections[buckets]

    def train(self, ages, targets, lr=0.001, epochs=200):
        ages_arr = np.array(ages, dtype=np.float32)
        tgt = np.array(targets, dtype=np.float32)
        # Normalize ages to prevent overflow
        age_scale = max(ages_arr.max(), 1.0)
        ages_norm = ages_arr / age_scale
        for _ in range(epochs):
            pred = self.bias_batch(ages)
            error = pred - tgt
            grad_slope = np.clip(2.0 * np.mean(error * ages_norm), -1.0, 1.0)
            grad_offset = np.clip(2.0 * np.mean(error), -1.0, 1.0)
            self._slope -= lr * grad_slope * age_scale
            self._offset -= lr * grad_offset
            buckets = np.array([self._get_bucket(a) for a in ages])
            for b in range(self._num_buckets):
                mask = buckets == b
                if mask.any():
                    grad_b = np.clip(2.0 * np.mean(error[mask]), -1.0, 1.0)
                    self._corrections[b] -= lr * grad_b
        loss = float(np.mean((self.bias_batch(ages) - tgt) ** 2))
        return {"loss": loss if not np.isnan(loss) else float("inf")}

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"slope": self._slope, "offset": self._offset,
                       "corrections": self._corrections.tolist()}, f)

    def load(self, path: str):
        with open(path) as f:
            d = json.load(f)
        self._slope = d["slope"]
        self._offset = d["offset"]
        self._corrections = np.array(d["corrections"], dtype=np.float32)
