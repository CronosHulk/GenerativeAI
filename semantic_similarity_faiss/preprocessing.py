from __future__ import annotations

import re


def preprocess_text(text: str) -> str:
    """Clean newsgroup-style text for semantic indexing."""
    text = re.sub(r"^From:.*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\S*@\S*\s?", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()
