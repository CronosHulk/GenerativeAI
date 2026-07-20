from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from embeddings import HashSentenceEncoder
from indexes import make_l2_index
from preprocessing import preprocess_text


@dataclass
class SearchResult:
    rank: int
    distance: float
    index: int
    original_text: str
    processed_text: str


class SemanticSearchEngine:
    def __init__(self, encoder=None, prefer_faiss: bool = True):
        self.encoder = encoder or HashSentenceEncoder()
        self.prefer_faiss = prefer_faiss
        self.documents: list[str] = []
        self.processed_documents: list[str] = []
        self.index = None

    def build(self, documents: Sequence[str]) -> None:
        self.documents = list(documents)
        self.processed_documents = [preprocess_text(document) for document in self.documents]
        vectors = self.encoder.encode(self.processed_documents)
        self.index = make_l2_index(vectors.shape[1], prefer_faiss=self.prefer_faiss)
        self.index.add(vectors)

    def search(self, query_text: str, k: int = 5) -> list[SearchResult]:
        if self.index is None:
            raise RuntimeError("build() must be called before search().")
        preprocessed_query = preprocess_text(query_text)
        query_vector = self.encoder.encode([preprocessed_query])
        distances, indices = self.index.search(query_vector, min(k, len(self.documents)))
        return [
            SearchResult(
                rank=rank,
                distance=float(distance),
                index=int(index),
                original_text=self.documents[int(index)],
                processed_text=self.processed_documents[int(index)],
            )
            for rank, (distance, index) in enumerate(zip(distances[0], indices[0]), start=1)
        ]
