"""Token counting service using the target LLM's tokenizer."""

from __future__ import annotations

from .config import Config


class TokenizerService:
    """Counts tokens using the tokenizer that matches the configured LLM."""

    def __init__(self, cfg: Config) -> None:
        self._model_name = cfg.llm_model
        self._tokenizer = None

    def _load_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, trust_remote_code=True
            )
        return self._tokenizer

    def count(self, text: str) -> int:
        """Return the number of tokens for the given text."""
        if not text or not text.strip():
            return 0
        tokenizer = self._load_tokenizer()
        return len(tokenizer.encode(text, add_special_tokens=False))


class DummyTokenizerService(TokenizerService):
    """
    Lightweight token counter for offline/CI testing.
    Approximates token count as words × 1.3 (typical subword expansion).
    """

    def __init__(self, cfg: Config | None = None) -> None:
        self._model_name = "dummy"
        self._tokenizer = None

    def _load_tokenizer(self):
        return None

    def count(self, text: str) -> int:
        """Approximate token count without a real tokenizer."""
        if not text or not text.strip():
            return 0
        # Rough approximation: ~1.3 tokens per whitespace-separated word
        words = text.split()
        return int(len(words) * 1.3)


def count_tokens(text: str, cfg: Config | None = None) -> int:
    """Module-level convenience for quick token counting."""
    if cfg is None:
        cfg = Config()
    svc = TokenizerService(cfg)
    return svc.count(text)
