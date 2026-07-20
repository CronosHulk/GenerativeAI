from __future__ import annotations

from llama_index.core.schema import Document, NodeWithScore

from retrievers import BM25Retriever, HybridRetriever, LocalVectorIndex, VectorStoreRetriever, node_text


class ProductionRAGPipeline:
    """Small production-style RAG skeleton with hybrid retrieval and citations."""

    def __init__(self, documents: list[Document], k: int = 4):
        self.documents = documents
        vector = VectorStoreRetriever(index=LocalVectorIndex(documents), k=k)
        keyword = BM25Retriever(documents, k=k)
        self.retriever = HybridRetriever(vector, keyword, k=k)

    def retrieve(self, question: str) -> list[NodeWithScore]:
        return self.retriever.retrieve(question)

    def answer(self, question: str) -> dict:
        sources = self.retrieve(question)
        context = "\n\n".join(node_text(source.node) for source in sources)
        return {
            "answer": (
                "Retrieved context for the question. Connect an LLM here to synthesize a final answer.\n\n"
                f"Question: {question}\n\nContext:\n{context}"
            ),
            "sources": [
                {
                    "title": source.node.metadata.get("title", "unknown"),
                    "score": source.score,
                    "text": node_text(source.node),
                }
                for source in sources
            ],
        }
