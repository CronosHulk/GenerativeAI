import math
import os
import re
from collections import Counter
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama


DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def make_ollama_chat(temperature: float = 0.2, model: str | None = None) -> ChatOllama:
    return ChatOllama(
        model=model or DEFAULT_OLLAMA_MODEL,
        base_url=DEFAULT_OLLAMA_URL,
        temperature=temperature,
    )


def print_section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def format_docs(docs: Iterable[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


class HashEmbeddings(Embeddings):
    """Small deterministic embedding model for local retrieval demos."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text.lower())
        counts = Counter(tokens)

        for token, count in counts.items():
            vector[hash(token) % self.dimensions] += float(count)

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
