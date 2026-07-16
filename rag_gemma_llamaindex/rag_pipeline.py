import argparse

from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

from settings import env, make_embeddings, make_llm, make_qdrant_client, wait_for_qdrant


QA_TEMPLATE = PromptTemplate(
    "You are a careful RAG assistant. Answer only from the provided context. "
    "If the answer is not in the context, say that the document does not "
    "contain enough information.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)


def build_index() -> VectorStoreIndex:
    wait_for_qdrant()
    vector_store = QdrantVectorStore(
        client=make_qdrant_client(),
        collection_name=env("QDRANT_COLLECTION", "llamaindex_state_of_union_rag"),
    )
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=make_embeddings(),
    )


def format_sources(source_nodes) -> str:
    lines = []
    for index, source_node in enumerate(source_nodes, start=1):
        node = source_node.node
        source = node.metadata.get("source", "unknown")
        chunk = node.metadata.get("chunk", "unknown")
        score = source_node.score
        preview = " ".join(node.get_content(metadata_mode="none").split())[:500]
        score_text = f" | score {score:.4f}" if score is not None else ""
        lines.append(f"[{index}] {source} | chunk {chunk}{score_text}\n{preview}")
    return "\n\n".join(lines)


def ask(question: str, k: int = 4) -> dict:
    index = build_index()
    query_engine = index.as_query_engine(
        llm=make_llm(),
        similarity_top_k=k,
        text_qa_template=QA_TEMPLATE,
    )
    response = query_engine.query(question)
    return {
        "answer": str(response),
        "source_nodes": response.source_nodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local RAG with Gemma, LlamaIndex and Qdrant.")
    parser.add_argument("question", help="Question for the RAG pipeline.")
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve.")
    parser.add_argument("--sources", action="store_true", help="Print retrieved source chunks.")

    args = parser.parse_args()

    result = ask(args.question, k=args.k)
    print(result["answer"])
    if args.sources:
        print("\nSources:\n")
        print(format_sources(result["source_nodes"]))


if __name__ == "__main__":
    main()
