import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import SAMPLE_DOCUMENTS
from embeddings import HashSentenceEncoder
from indexes import NumpyL2Index
from preprocessing import preprocess_text
from search_engine import SemanticSearchEngine


class SemanticSearchTests(unittest.TestCase):
    def test_preprocess_removes_email_noise(self):
        text = "From: user@example.com\nHello, NASA 123!!!"
        self.assertEqual(preprocess_text(text), "hello nasa")

    def test_numpy_l2_index_returns_nearest_vector(self):
        index = NumpyL2Index(2)
        index.add([[0, 0], [10, 10], [1, 1]])
        distances, indices = index.search([[0.2, 0.2]], k=2)
        self.assertEqual(indices[0].tolist(), [0, 2])
        self.assertLess(distances[0][0], distances[0][1])

    def test_search_finds_motorcycle_document(self):
        engine = SemanticSearchEngine(encoder=HashSentenceEncoder(), prefer_faiss=False)
        engine.build(SAMPLE_DOCUMENTS)
        results = engine.search("motorcycle helmet tires", k=1)
        self.assertIn("Motorcycle riders", results[0].original_text)

    def test_search_requires_build(self):
        engine = SemanticSearchEngine()
        with self.assertRaises(RuntimeError):
            engine.search("motorcycle")


if __name__ == "__main__":
    unittest.main()
