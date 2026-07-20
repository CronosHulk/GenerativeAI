import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import MOVIES, PARENT_TEXTS, POLICIES, REFERENCE_DOCS, TECH_DOCS
from pipeline import ProductionRAGPipeline
from retrievers import (
    AutoMergingRetriever,
    BM25Retriever,
    DocumentSummaryRetriever,
    HybridRetriever,
    LocalVectorIndex,
    ParentDocumentRetriever,
    QueryFusionRetriever,
    RecursiveRetriever,
    SelfQueryRetriever,
    VectorStoreRetriever,
)


class RetrieverTests(unittest.TestCase):
    def test_similarity_finds_smoking(self):
        nodes = VectorStoreRetriever(index=LocalVectorIndex(POLICIES), k=1).retrieve("smoking vaping")
        self.assertEqual(nodes[0].node.metadata["section"], "safety")

    def test_self_query_filters_rating(self):
        nodes = SelfQueryRetriever(index=LocalVectorIndex(MOVIES)).retrieve("movie rated above 8.5")
        self.assertTrue(nodes)
        self.assertTrue(all(node.node.metadata["rating"] > 8.5 for node in nodes))

    def test_self_query_filters_director(self):
        nodes = SelfQueryRetriever(index=LocalVectorIndex(MOVIES)).retrieve("dream by Christopher Nolan")
        self.assertEqual([node.node.metadata["title"] for node in nodes], ["Inception"])

    def test_parent_returns_large_context(self):
        nodes = ParentDocumentRetriever.from_documents(PARENT_TEXTS).retrieve("outdoor smoking area")
        self.assertEqual(nodes[0].node.metadata["title"], "Safety handbook")
        self.assertIn("Fire exits", nodes[0].node.get_content(metadata_mode="none"))

    def test_bm25_finds_keyword_doc(self):
        nodes = BM25Retriever(TECH_DOCS, k=1).retrieve("BM25 exact keywords")
        self.assertEqual(nodes[0].node.metadata["doc_id"], "bm25")

    def test_document_summary_returns_full_doc(self):
        nodes = DocumentSummaryRetriever(TECH_DOCS, k=1).retrieve("summary retrieval original full document")
        self.assertEqual(nodes[0].node.metadata["doc_id"], "summary")
        self.assertIn("original full document", nodes[0].node.get_content(metadata_mode="none"))

    def test_auto_merging_returns_parent_when_many_children_match(self):
        nodes = AutoMergingRetriever.from_documents(PARENT_TEXTS).retrieve("smoking vaping indoor outdoor fire exits")
        self.assertEqual(nodes[0].node.metadata["title"], "Safety handbook")

    def test_recursive_follows_references(self):
        references = {doc.metadata["doc_id"]: doc for doc in REFERENCE_DOCS}
        base = VectorStoreRetriever(LocalVectorIndex(REFERENCE_DOCS), k=1)
        nodes = RecursiveRetriever(base, references).retrieve("overview semantic exact matching")
        titles = [node.node.metadata["title"] for node in nodes]
        self.assertIn("Vector Details", titles)
        self.assertIn("BM25 Details", titles)

    def test_query_fusion_combines_retrievers(self):
        vector = VectorStoreRetriever(LocalVectorIndex(TECH_DOCS), k=2)
        keyword = BM25Retriever(TECH_DOCS, k=2)
        nodes = QueryFusionRetriever([vector, keyword], query_generator=lambda query: [], k=2).retrieve("BM25 semantic")
        self.assertEqual(len(nodes), 2)

    def test_hybrid_retriever_uses_keyword_and_vector(self):
        vector = VectorStoreRetriever(LocalVectorIndex(TECH_DOCS), k=3)
        keyword = BM25Retriever(TECH_DOCS, k=3)
        nodes = HybridRetriever(vector, keyword, k=3).retrieve("BM25 exact keywords")
        self.assertEqual(nodes[0].node.metadata["doc_id"], "bm25")

    def test_production_pipeline_returns_sources(self):
        result = ProductionRAGPipeline(TECH_DOCS, k=2).answer("BM25 keyword search")
        self.assertTrue(result["sources"])
        self.assertIn("Context:", result["answer"])


if __name__ == "__main__":
    unittest.main()
