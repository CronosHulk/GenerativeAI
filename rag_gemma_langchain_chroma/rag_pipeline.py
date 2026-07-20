import argparse
from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from settings import make_chroma_collection, make_llm, wait_for_chroma


def format_docs(documents: list[Document]) -> str:
    formatted = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        chunk = doc.metadata.get("chunk", "unknown")
        formatted.append(f"Source: {source} | chunk: {chunk}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def format_sources(documents: list[Document]) -> str:
    lines = []
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk = doc.metadata.get("chunk", "unknown")
        preview = " ".join(doc.page_content.split())[:500]
        lines.append(f"[{index}] {source} | chunk {chunk}\n{preview}")
    return "\n\n".join(lines)


def retrieve_documents(question: str, k: int) -> list[Document]:
    collection = make_chroma_collection()
    result = collection.query(query_texts=[question], n_results=k)
    documents = result.get("documents") or [[]]
    metadatas = result.get("metadatas") or [[]]

    return [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(documents[0], metadatas[0], strict=False)
    ]


def build_rag_chain(k: int = 4):
    wait_for_chroma()
    retriever = RunnableLambda(lambda question: retrieve_documents(question, k=k))

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful RAG assistant. Answer only from the provided "
                "context. If the answer is not in the context, say that the "
                "document does not contain enough information. Keep the answer "
                "concise and cite source chunk numbers when useful.",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion:\n{question}",
            ),
        ]
    )

    with_context = RunnableParallel(
        {
            "question": RunnablePassthrough(),
            "source_documents": retriever,
        }
    ) | RunnablePassthrough.assign(
        context=RunnableLambda(lambda values: format_docs(values["source_documents"]))
    )

    return with_context | {
        "answer": prompt | make_llm() | StrOutputParser(),
        "source_documents": itemgetter("source_documents"),
    }


def ask(question: str, k: int = 4) -> dict:
    return build_rag_chain(k=k).invoke(question)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local RAG with Gemma, LangChain and Chroma.")
    parser.add_argument("question", help="Question for the RAG chain.")
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve.")
    parser.add_argument("--sources", action="store_true", help="Print retrieved source chunks.")

    args = parser.parse_args()

    result = ask(args.question, k=args.k)
    print(result["answer"])
    if args.sources:
        print("\nSources:\n")
        print(format_sources(result["source_documents"]))


if __name__ == "__main__":
    main()
