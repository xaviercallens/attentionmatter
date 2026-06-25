"""Content extractors that convert multi-modal content to embeddable text."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any


class ContentExtractor(ABC):
    """Base class for modality-specific text extraction."""

    @abstractmethod
    def extract_text(self, content: Any, metadata: dict | None = None) -> str:
        """Extract embeddable text representation from raw content."""
        ...

    @abstractmethod
    def estimate_tokens(self, content: Any) -> int:
        """Estimate token count for the content."""
        ...


class CodeExtractor(ContentExtractor):
    """
    Extracts embeddable text from code snippets.
    Focuses on function/class names, docstrings, comments,
    and key identifiers for semantic matching.
    """

    def extract_text(self, content: Any, metadata: dict | None = None) -> str:
        code = str(content)
        parts = []

        # Extract docstrings
        docstrings = re.findall(r'"""(.*?)"""', code, re.DOTALL)
        docstrings += re.findall(r"'''(.*?)'''", code, re.DOTALL)
        parts.extend(ds.strip()[:200] for ds in docstrings)

        # Extract function/class names
        funcs = re.findall(r'(?:def|class|function|const|let|var)\s+(\w+)', code)
        if funcs:
            parts.append("Defines: " + ", ".join(funcs[:10]))

        # Extract comments
        comments = re.findall(r'#\s*(.+)$|//\s*(.+)$', code, re.MULTILINE)
        for groups in comments[:5]:
            c = next((g for g in groups if g), "")
            if c:
                parts.append(c.strip())

        # Extract imports (indicate what the code uses)
        imports = re.findall(r'(?:import|from|require|include)\s+(.+?)(?:\s|;|$)', code)
        if imports:
            parts.append("Uses: " + ", ".join(imports[:5]))

        # Fallback: first meaningful line
        if not parts:
            lines = [l.strip() for l in code.split("\n") if l.strip() and not l.strip().startswith("#")]
            parts.append(lines[0][:100] if lines else "code snippet")

        lang = (metadata or {}).get("language", "")
        prefix = f"[{lang} code] " if lang else "[code] "
        return prefix + " | ".join(parts)

    def estimate_tokens(self, content: Any) -> int:
        code = str(content)
        # Code typically has more tokens per line than prose
        return int(len(code.split()) * 1.5)


class TableExtractor(ContentExtractor):
    """
    Extracts embeddable text from tabular data.
    Captures column names, sample values, and summary stats.
    """

    def extract_text(self, content: Any, metadata: dict | None = None) -> str:
        parts = ["[table]"]

        if isinstance(content, dict):
            columns = content.get("columns", [])
            rows = content.get("rows", [])
        elif isinstance(content, list) and content:
            if isinstance(content[0], dict):
                columns = list(content[0].keys())
                rows = content
            else:
                columns = [f"col_{i}" for i in range(len(content[0]))]
                rows = content
        else:
            return "[table] empty table"

        if columns:
            parts.append(f"Columns: {', '.join(str(c) for c in columns)}")

        # Sample values from first few rows
        if rows:
            parts.append(f"Rows: {len(rows)}")
            sample = rows[0]
            if isinstance(sample, dict):
                vals = [f"{k}={v}" for k, v in list(sample.items())[:5]]
            else:
                vals = [str(v) for v in sample[:5]]
            parts.append(f"Sample: {', '.join(vals)}")

        return " | ".join(parts)

    def estimate_tokens(self, content: Any) -> int:
        if isinstance(content, dict):
            rows = content.get("rows", [])
            cols = content.get("columns", [])
        elif isinstance(content, list):
            rows = content
            cols = list(content[0].keys()) if content and isinstance(content[0], dict) else []
        else:
            return 10
        # ~3 tokens per cell + header
        return len(cols) * (len(rows) + 1) * 3


class ImageExtractor(ContentExtractor):
    """
    Extracts embeddable text from image metadata/descriptions.
    Uses alt text, captions, OCR text, or file metadata.
    """

    def extract_text(self, content: Any, metadata: dict | None = None) -> str:
        meta = metadata or {}
        parts = ["[image]"]

        if meta.get("caption"):
            parts.append(meta["caption"])
        if meta.get("alt_text"):
            parts.append(meta["alt_text"])
        if meta.get("ocr_text"):
            parts.append(f"Text in image: {meta['ocr_text'][:200]}")
        if meta.get("description"):
            parts.append(meta["description"])
        if meta.get("filename"):
            parts.append(f"File: {meta['filename']}")

        if len(parts) == 1:
            parts.append("image content (no description available)")

        return " | ".join(parts)

    def estimate_tokens(self, content: Any) -> int:
        # Images are referenced, not inlined — small token cost
        return 20


class StructuredDataExtractor(ContentExtractor):
    """
    Extracts embeddable text from structured data (JSON, dicts, configs).
    Captures key names, types, and sample values.
    """

    def extract_text(self, content: Any, metadata: dict | None = None) -> str:
        parts = ["[structured data]"]

        if isinstance(content, dict):
            keys = list(content.keys())[:10]
            parts.append(f"Keys: {', '.join(str(k) for k in keys)}")
            # Sample key-value pairs
            samples = []
            for k in keys[:5]:
                v = content[k]
                if isinstance(v, (str, int, float, bool)):
                    samples.append(f"{k}={v}")
                elif isinstance(v, list):
                    samples.append(f"{k}=[{len(v)} items]")
                elif isinstance(v, dict):
                    samples.append(f"{k}={{...}}")
            if samples:
                parts.append(f"Values: {', '.join(samples)}")
        elif isinstance(content, str):
            # Try to describe the structure
            parts.append(content[:200])
        else:
            parts.append(str(content)[:200])

        schema = (metadata or {}).get("schema_type", "")
        if schema:
            parts[0] = f"[{schema}]"

        return " | ".join(parts)

    def estimate_tokens(self, content: Any) -> int:
        if isinstance(content, dict):
            return int(len(str(content)) / 4)
        return int(len(str(content)) / 4)
