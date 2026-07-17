from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from typing import Any, Callable, Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever


TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)


class LocalHashEmbeddings(Embeddings):
    """Small deterministic embeddings for offline demonstrations."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            slot = int.from_bytes(digest, "big") % self.dimensions
            vector[slot] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class LocalVectorIndex:
    def __init__(self, documents: Iterable[Document], embeddings: Embeddings | None = None):
        self.documents = list(documents)
        self.embeddings = embeddings or LocalHashEmbeddings()
        self.vectors = self.embeddings.embed_documents([doc.page_content for doc in self.documents])

    def scored(self, query: str) -> list[tuple[Document, float]]:
        query_vector = self.embeddings.embed_query(query)
        pairs = [(doc, cosine(query_vector, vector)) for doc, vector in zip(self.documents, self.vectors)]
        return sorted(pairs, key=lambda item: item[1], reverse=True)

    def similarity(self, query: str, k: int = 4, threshold: float | None = None) -> list[Document]:
        pairs = self.scored(query)
        if threshold is not None:
            pairs = [pair for pair in pairs if pair[1] >= threshold]
        return [doc for doc, _ in pairs[:k]]

    def mmr(self, query: str, k: int = 4, fetch_k: int = 10, lambda_mult: float = 0.5) -> list[Document]:
        candidates = self.scored(query)[:fetch_k]
        selected: list[tuple[Document, list[float]]] = []
        while candidates and len(selected) < k:
            best_position, best_score = 0, float("-inf")
            for position, (doc, relevance) in enumerate(candidates):
                vector = self.embeddings.embed_query(doc.page_content)
                redundancy = max((cosine(vector, chosen_vector) for _, chosen_vector in selected), default=0.0)
                score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
                if score > best_score:
                    best_position, best_score = position, score
            doc, _ = candidates.pop(best_position)
            selected.append((doc, self.embeddings.embed_query(doc.page_content)))
        return [doc for doc, _ in selected]


class VectorStoreRetriever(BaseRetriever):
    index: LocalVectorIndex
    search_type: str = "similarity"
    k: int = 4
    threshold: float | None = None

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        if self.search_type == "mmr":
            return self.index.mmr(query, k=self.k)
        if self.search_type == "similarity_score_threshold":
            return self.index.similarity(query, k=self.k, threshold=self.threshold)
        return self.index.similarity(query, k=self.k)


class MultiQueryRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    query_generator: Callable[[str], list[str]]

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        unique: dict[tuple[str, tuple], Document] = {}
        for variant in [query, *self.query_generator(query)]:
            for doc in self.base_retriever.invoke(variant):
                key = (doc.page_content, tuple(sorted(doc.metadata.items())))
                unique[key] = doc
        return list(unique.values())


def local_query_variants(query: str) -> list[str]:
    """Rule-based stand-in for the LLM normally used by MultiQueryRetriever."""
    lowered = query.lower()
    variants = [f"rules and requirements about {query}", f"company handbook {query}"]
    synonyms = {"smoking": "vaping tobacco outdoor area", "email": "messages communication confidential data", "remote": "work from home manager approval"}
    variants.extend(value for key, value in synonyms.items() if key in lowered)
    return variants


class SelfQueryRetriever(BaseRetriever):
    index: LocalVectorIndex
    k: int = 4
    planner: Callable[[str], Any] | None = None

    class Config:
        arbitrary_types_allowed = True

    def _parse(self, query: str) -> tuple[str, list[Callable[[dict], bool]]]:
        filters: list[Callable[[dict], bool]] = []
        lowered = query.lower()
        rating = re.search(r"(?:rating|rated|рейтинг\w*)\s*(?:higher than|above|over|выше)?\s*(\d+(?:\.\d+)?)", lowered)
        if rating:
            value = float(rating.group(1))
            filters.append(lambda meta, value=value: float(meta.get("rating", 0)) > value)
        year = re.search(r"(?:after|после)\s*(\d{4})", lowered)
        if year:
            value = int(year.group(1))
            filters.append(lambda meta, value=value: int(meta.get("year", 0)) > value)
        for field in ("director", "genre"):
            values = {str(doc.metadata.get(field, "")) for doc in self.index.documents}
            for value in values:
                if value and value.lower() in lowered:
                    filters.append(lambda meta, field=field, value=value: meta.get(field) == value)
        semantic = re.sub(r"(?:rating|rated|рейтинг\w*)[^,?.]*|(?:after|после)\s*\d{4}", "", query, flags=re.I).strip(" ,?.")
        return semantic, filters

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        if self.planner is None:
            semantic, filters = self._parse(query)
        else:
            plan = self.planner(query)
            semantic = plan.semantic_query
            filters = []
            if plan.min_rating is not None:
                filters.append(lambda meta, value=plan.min_rating: float(meta.get("rating", 0)) > value)
            if plan.after_year is not None:
                filters.append(lambda meta, value=plan.after_year: int(meta.get("year", 0)) > value)
            if plan.director:
                filters.append(lambda meta, value=plan.director.lower(): str(meta.get("director", "")).lower() == value)
            if plan.genre:
                filters.append(lambda meta, value=plan.genre.lower(): str(meta.get("genre", "")).lower() == value)
        candidates = self.index.scored(semantic or query)
        return [doc for doc, _ in candidates if all(rule(doc.metadata) for rule in filters)][: self.k]


def split_words(text: str, size: int = 12, overlap: int = 3) -> list[str]:
    words = text.split()
    step = max(1, size - overlap)
    return [" ".join(words[start:start + size]) for start in range(0, len(words), step) if words[start:start + size]]


class ParentDocumentRetriever(BaseRetriever):
    child_index: LocalVectorIndex
    parents: dict[str, Document]
    k_children: int = 4

    @classmethod
    def from_documents(cls, documents: Iterable[Document], child_size: int = 12, overlap: int = 3):
        parents: dict[str, Document] = {}
        children: list[Document] = []
        for number, parent in enumerate(documents):
            parent_id = f"parent-{number}"
            parents[parent_id] = parent
            for chunk in split_words(parent.page_content, child_size, overlap):
                children.append(Document(page_content=chunk, metadata={"parent_id": parent_id}))
        return cls(child_index=LocalVectorIndex(children), parents=parents)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        result: list[Document] = []
        seen: set[str] = set()
        for child in self.child_index.similarity(query, k=self.k_children):
            parent_id = child.metadata["parent_id"]
            if parent_id not in seen:
                result.append(self.parents[parent_id])
                seen.add(parent_id)
        return result
