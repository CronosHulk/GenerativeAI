from __future__ import annotations

import sys

from data import MOVIES, PARENT_TEXTS, POLICIES, REFERENCE_DOCS, TECH_DOCS
from ollama_gemma import make_gemma, make_multi_query_generator, make_self_query_planner
from pipeline import ProductionRAGPipeline
from retrievers import (
    AutoMergingRetriever,
    BM25Retriever,
    DocumentSummaryRetriever,
    HybridRetriever,
    LocalVectorIndex,
    MultiQueryRetriever,
    ParentDocumentRetriever,
    QueryFusionRetriever,
    RecursiveRetriever,
    SelfQueryRetriever,
    VectorStoreRetriever,
    local_query_variants,
    node_text,
)


def show(title, nodes):
    print(f"\n=== {title} ===")
    for number, node_with_score in enumerate(nodes, 1):
        node = node_with_score.node
        score = node_with_score.score
        score_text = f" score={score:.4f}" if score is not None else ""
        print(f"{number}. {node_text(node)}\n   metadata={node.metadata}{score_text}")


def vector_demo():
    index = LocalVectorIndex(POLICIES)
    show("Similarity: smoking policy", VectorStoreRetriever(index=index, k=2).retrieve("smoking vaping policy"))
    show("MMR: company policy", VectorStoreRetriever(index=index, search_type="mmr", k=3).retrieve("company policy rules"))
    show("Threshold >= 0.20", VectorStoreRetriever(index=index, search_type="similarity_score_threshold", k=5, threshold=0.20).retrieve("email confidential data"))


def bm25_demo():
    retriever = BM25Retriever(TECH_DOCS, k=2)
    show("BM25 keyword search", retriever.retrieve("BM25 exact keywords product codes"))


def summary_demo():
    retriever = DocumentSummaryRetriever(TECH_DOCS, k=2)
    show("Document summary selection", retriever.retrieve("select full documents using concise summaries"))


def auto_merge_demo():
    retriever = AutoMergingRetriever.from_documents(PARENT_TEXTS)
    show("Auto merging child hits into parent context", retriever.retrieve("smoking vaping indoor outdoor fire exits"))


def recursive_demo():
    index = LocalVectorIndex(REFERENCE_DOCS)
    base = VectorStoreRetriever(index=index, k=1)
    references = {doc.metadata["doc_id"]: doc for doc in REFERENCE_DOCS}
    retriever = RecursiveRetriever(base, references)
    show("Recursive reference following", retriever.retrieve("overview semantic exact matching"))


def fusion_demo():
    vector = VectorStoreRetriever(index=LocalVectorIndex(TECH_DOCS), k=3)
    keyword = BM25Retriever(TECH_DOCS, k=3)
    for mode in ("rrf", "relative_score", "distribution"):
        retriever = QueryFusionRetriever([vector, keyword], query_generator=local_query_variants, mode=mode, k=3)
        show(f"Query fusion: {mode}", retriever.retrieve("semantic and exact keyword retrieval"))


def hybrid_demo():
    vector = VectorStoreRetriever(index=LocalVectorIndex(TECH_DOCS), k=3)
    keyword = BM25Retriever(TECH_DOCS, k=3)
    retriever = HybridRetriever(vector, keyword, k=3)
    show("Custom hybrid retriever", retriever.retrieve("BM25 semantic search embeddings"))


def production_demo():
    pipeline = ProductionRAGPipeline(TECH_DOCS, k=3)
    result = pipeline.answer("When should I use BM25 instead of vector search?")
    print("\n=== Production RAG pipeline ===")
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source['title']} score={source['score']:.4f}")


def multi_demo():
    base = VectorStoreRetriever(index=LocalVectorIndex(POLICIES), k=1)
    retriever = MultiQueryRetriever(base_retriever=base, query_generator=make_multi_query_generator(make_gemma()))
    show("Multi-query union", retriever.retrieve("smoking policy"))


def self_demo():
    retriever = SelfQueryRetriever(index=LocalVectorIndex(MOVIES), planner=make_self_query_planner(make_gemma()))
    show("Metadata filter", retriever.retrieve("movie rated above 8.5"))
    show("Semantic query + director filter", retriever.retrieve("dream movie by Christopher Nolan"))
    show("Composite filters", retriever.retrieve("science fiction rated above 8.5"))


def parent_demo():
    retriever = ParentDocumentRetriever.from_documents(PARENT_TEXTS)
    show("Parent context for a child match", retriever.retrieve("outdoor smoking area"))


DEMOS = {
    "vector": vector_demo,
    "bm25": bm25_demo,
    "summary": summary_demo,
    "auto_merge": auto_merge_demo,
    "recursive": recursive_demo,
    "fusion": fusion_demo,
    "hybrid": hybrid_demo,
    "production": production_demo,
    "multi": multi_demo,
    "self": self_demo,
    "parent": parent_demo,
}

if __name__ == "__main__":
    selected = sys.argv[1:] or list(DEMOS)
    for name in selected:
        DEMOS[name]()
