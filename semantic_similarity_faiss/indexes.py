from __future__ import annotations

import numpy as np


class NumpyL2Index:
    """Tiny FAISS-like IndexFlatL2 fallback with add/search methods."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = np.empty((0, dimension), dtype="float32")

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"expected vectors with shape (n, {self.dimension})")
        self.vectors = np.vstack([self.vectors, vectors])

    def search(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        query_vectors = np.asarray(query_vectors, dtype="float32")
        distances = ((query_vectors[:, None, :] - self.vectors[None, :, :]) ** 2).sum(axis=2)
        top = np.argsort(distances, axis=1)[:, :k]
        top_distances = np.take_along_axis(distances, top, axis=1)
        return top_distances.astype("float32"), top.astype("int64")


def make_l2_index(dimension: int, prefer_faiss: bool = True):
    if prefer_faiss:
        try:
            import faiss

            return faiss.IndexFlatL2(dimension)
        except ImportError:
            pass
    return NumpyL2Index(dimension)
