"""Multi-modal data types and candidate models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ModalityType(Enum):
    """Supported content modalities."""
    TEXT = "text"
    CODE = "code"
    TABLE = "table"
    IMAGE = "image"
    STRUCTURED = "structured"  # JSON/dict data


@dataclass
class MultiModalCandidate:
    """
    A candidate that can represent different content modalities.

    Each modality carries its raw content plus a text representation
    used for embedding and scoring. The text_repr is what gets embedded;
    the raw_content is what gets included in the final prompt.
    """
    text_repr: str  # text used for embedding/scoring
    raw_content: Any  # original content (code str, table dict, image path, etc.)
    modality: ModalityType = ModalityType.TEXT
    age: int = 0
    turn: int = -1
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None

    # Modality-specific fields
    language: str = ""  # for code: python, javascript, etc.
    columns: list[str] = field(default_factory=list)  # for tables
    image_description: str = ""  # for images: alt text or caption
    schema_type: str = ""  # for structured: json, yaml, etc.

    @property
    def prompt_text(self) -> str:
        """Text to include in the LLM prompt (formatted per modality)."""
        if self.modality == ModalityType.CODE:
            lang = self.language or ""
            return f"```{lang}\n{self.raw_content}\n```"
        elif self.modality == ModalityType.TABLE:
            return self._format_table()
        elif self.modality == ModalityType.IMAGE:
            return f"[Image: {self.image_description or 'no description'}]"
        elif self.modality == ModalityType.STRUCTURED:
            return f"```{self.schema_type}\n{self.raw_content}\n```"
        return str(self.raw_content)

    def _format_table(self) -> str:
        """Format table data as markdown."""
        if isinstance(self.raw_content, dict):
            rows = self.raw_content.get("rows", [])
            cols = self.columns or self.raw_content.get("columns", [])
        elif isinstance(self.raw_content, list):
            rows = self.raw_content
            cols = self.columns
        else:
            return str(self.raw_content)

        if not cols and rows:
            cols = [f"col_{i}" for i in range(len(rows[0]))]

        lines = ["| " + " | ".join(str(c) for c in cols) + " |"]
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for row in rows[:20]:  # cap at 20 rows for prompt
            if isinstance(row, dict):
                vals = [str(row.get(c, "")) for c in cols]
            else:
                vals = [str(v) for v in row]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)
