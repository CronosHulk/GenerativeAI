import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import MOVIES, PARENT_TEXTS, POLICIES
from retrievers import LocalVectorIndex, ParentDocumentRetriever, SelfQueryRetriever, VectorStoreRetriever


class RetrieverTests(unittest.TestCase):
    def test_similarity_finds_smoking(self):
        docs = VectorStoreRetriever(index=LocalVectorIndex(POLICIES), k=1).invoke("smoking vaping")
        self.assertEqual(docs[0].metadata["section"], "safety")

    def test_self_query_filters_rating(self):
        docs = SelfQueryRetriever(index=LocalVectorIndex(MOVIES)).invoke("movie rated above 8.5")
        self.assertTrue(docs)
        self.assertTrue(all(doc.metadata["rating"] > 8.5 for doc in docs))

    def test_self_query_filters_director(self):
        docs = SelfQueryRetriever(index=LocalVectorIndex(MOVIES)).invoke("dream by Christopher Nolan")
        self.assertEqual([doc.metadata["title"] for doc in docs], ["Inception"])

    def test_parent_returns_large_context(self):
        docs = ParentDocumentRetriever.from_documents(PARENT_TEXTS).invoke("outdoor smoking area")
        self.assertEqual(docs[0].metadata["title"], "Safety handbook")
        self.assertIn("Fire exits", docs[0].page_content)


if __name__ == "__main__":
    unittest.main()
