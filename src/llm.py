"""LLM client wrapper for local model generation."""

from __future__ import annotations

from .config import Config


class LLMClient:
    """Wraps a local Hugging Face causal LM with optional 4-bit quantization."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._model = None
        self._tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._cfg.llm_model, trust_remote_code=True
            )

            load_kwargs: dict = {
                "device_map": "auto",
                "trust_remote_code": True,
            }

            if self._cfg.use_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs["quantization_config"] = bnb_config
            else:
                load_kwargs["torch_dtype"] = torch.float16

            self._model = AutoModelForCausalLM.from_pretrained(
                self._cfg.llm_model, **load_kwargs
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load LLM '{self._cfg.llm_model}'. "
                f"Ensure the model is downloadable and sufficient GPU memory is available. "
                f"Try use_4bit=True or a smaller model. Error: {e}"
            ) from e

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """Generate text from a prompt. Returns only the newly generated tokens."""
        import torch

        max_tokens = max_new_tokens or self._cfg.max_new_tokens
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][input_len:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


class DummyLLMClient:
    """
    A deterministic stub LLM for fast testing without GPU.
    Echoes back any text that looks like a key fact from the context.
    """

    name: str = "DummyLLM"

    def __init__(self, cfg: Config | None = None) -> None:
        self._cfg = cfg

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """
        Simple heuristic: search for patterns that look like codes, numbers, or
        key facts in the prompt context and echo them back in a short answer.
        """
        # Extract lines from the [Context] section
        lines = prompt.split("\n")
        context_lines = []
        in_context = False
        for line in lines:
            if line.strip().startswith("[Context]"):
                in_context = True
                continue
            if line.strip().startswith("[User Query]"):
                in_context = False
                continue
            if in_context and line.strip():
                context_lines.append(line.strip())

        # Look for key data patterns in context (codes, numbers, names)
        import re
        facts = []
        for line in context_lines:
            # Find alphanumeric codes (3+ chars, uppercase with digits)
            codes = re.findall(r'\b[A-Z][A-Z0-9]{2,}(?:-[A-Z0-9]+)*\b', line)
            facts.extend(codes)
            # Find codes/numbers with dashes (like ACC-9182736, 555-0142)
            dashed = re.findall(r'\b[A-Z0-9]+-[A-Z0-9]+\b', line)
            facts.extend(dashed)
            # Find phone numbers
            phones = re.findall(r'\b\d{3}-\d{4}\b', line)
            facts.extend(phones)
            # Find hex-style codes (0x...)
            hex_codes = re.findall(r'0x[0-9A-Fa-f]+', line)
            facts.extend(hex_codes)
            # Find time patterns (e.g., "6 PM", "8 AM")
            times = re.findall(r'\b\d{1,2}\s*(?:AM|PM|am|pm)\b', line)
            facts.extend(times)
            # Find quoted strings
            quoted = re.findall(r'"([^"]+)"', line)
            facts.extend(quoted)
            # Find explicit key-value patterns
            kv = re.findall(r'(?:code|number|reference|name|preference|address|phone|closes?)[:\s]+([^\.,\n]+)', line, re.IGNORECASE)
            facts.extend(kv)
            # Find standalone important words (capitalized proper nouns)
            if "name" in line.lower() or "Name" in line:
                names = re.findall(r'\b[A-Z][a-z]{2,}\b', line)
                # Filter out common words
                common = {"The", "This", "That", "Your", "Here", "Turn", "Memory", "User", "Sure"}
                facts.extend([n for n in names if n not in common])
            # Find "vegetarian", "vegan" etc
            dietary = re.findall(r'\b(?:vegetarian|vegan|pescatarian|gluten-free)\b', line, re.IGNORECASE)
            facts.extend(dietary)

        if facts:
            # Return a response containing the found facts
            unique_facts = list(dict.fromkeys(facts))[:8]
            return f"Based on the available information: {', '.join(unique_facts)}"

        # Fallback: echo the last context line if any
        if context_lines:
            return f"Based on the context: {context_lines[-1]}"

        return "I don't have enough context to answer that question."
