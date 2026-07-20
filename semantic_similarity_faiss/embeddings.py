from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable

import numpy as np


TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)


class HashSentenceEncoder:
    """Small deterministic encoder for offline demos and tests.

    It is not a replacement for Universal Sentence Encoder quality, but keeps the
    lab runnable when TensorFlow Hub is not installed.
    """

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return np.vstack([self._embed(text) for text in texts]).astype("float32")

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype="float32")
        for token in TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            slot = int.from_bytes(digest, "big") % self.dimensions
            vector[slot] += 1.0
        norm = math.sqrt(float(np.dot(vector, vector))) or 1.0
        return vector / norm


class UniversalSentenceEncoder:
    """TensorFlow Hub Universal Sentence Encoder wrapper."""

    def __init__(self, url: str = "https://tfhub.dev/google/universal-sentence-encoder/4"):
        try:
            import tensorflow_hub as hub
        except ImportError as exc:
            raise RuntimeError(
                "tensorflow-hub is required for UniversalSentenceEncoder. "
                "Install requirements.txt or use HashSentenceEncoder."
            ) from exc
        self.model = hub.load(url)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return self.model(list(texts)).numpy().astype("float32")
