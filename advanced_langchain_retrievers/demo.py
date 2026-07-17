from __future__ import annotations

import sys

from data import MOVIES, PARENT_TEXTS, POLICIES
from retrievers import (
    LocalVectorIndex,
    MultiQueryRetriever,
    ParentDocumentRetriever,
    SelfQueryRetriever,
    VectorStoreRetriever,
)
from ollama_gemma import make_gemma, make_multi_query_generator, make_self_query_planner


def show(title, documents):
    print(f"\n=== {title} ===")
    for number, doc in enumerate(documents, 1):
        print(f"{number}. {doc.page_content}\n   metadata={doc.metadata}")


def vector_demo():
    index = LocalVectorIndex(POLICIES)
    show("Similarity: smoking policy", VectorStoreRetriever(index=index, k=2).invoke("smoking vaping policy"))
    show("MMR: company policy", VectorStoreRetriever(index=index, search_type="mmr", k=3).invoke("company policy rules"))
    show("Threshold >= 0.20", VectorStoreRetriever(index=index, search_type="similarity_score_threshold", k=5, threshold=0.20).invoke("email confidential data"))


def multi_demo():
    base = VectorStoreRetriever(index=LocalVectorIndex(POLICIES), k=1)
    retriever = MultiQueryRetriever(base_retriever=base, query_generator=make_multi_query_generator(make_gemma()))
    show("Multi-query union", retriever.invoke("smoking policy"))


def self_demo():
    retriever = SelfQueryRetriever(index=LocalVectorIndex(MOVIES), planner=make_self_query_planner(make_gemma()))
    show("Metadata filter", retriever.invoke("movie rated above 8.5"))
    show("Semantic query + director filter", retriever.invoke("dream movie by Christopher Nolan"))
    show("Composite filters", retriever.invoke("science fiction rated above 8.5"))


def parent_demo():
    retriever = ParentDocumentRetriever.from_documents(PARENT_TEXTS)
    show("Parent context for a child match", retriever.invoke("outdoor smoking area"))


DEMOS = {"vector": vector_demo, "multi": multi_demo, "self": self_demo, "parent": parent_demo}

if __name__ == "__main__":
    selected = sys.argv[1:] or list(DEMOS)
    for name in selected:
        DEMOS[name]()
