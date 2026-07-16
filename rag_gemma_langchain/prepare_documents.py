import argparse
from pathlib import Path

import requests
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import (
    DATA_DIR,
    DEFAULT_SOURCE_URL,
    env,
    make_embeddings,
    make_qdrant_client,
    wait_for_qdrant,
)


def download_document(
    url: str = env("RAG_SOURCE_URL", DEFAULT_SOURCE_URL),
    filename: str = "stateOfUnion.txt",
) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    path.write_text(response.content.decode("utf-8"), encoding="utf-8")
    return path


def load_document(path: Path | None = None, url: str | None = None) -> list[Document]:
    if path is None:
        path = DATA_DIR / "stateOfUnion.txt"
    if not path.exists():
        path = download_document(url=url or env("RAG_SOURCE_URL", DEFAULT_SOURCE_URL))

    text = path.read_text(encoding="utf-8")
    return [
        Document(
            page_content=text,
            metadata={
                "source": str(path),
                "url": env("RAG_SOURCE_URL", DEFAULT_SOURCE_URL),
            },
        )
    ]


def prepare_document(url: str | None = None) -> list[Document]:
    path = download_document(url=url or env("RAG_SOURCE_URL", DEFAULT_SOURCE_URL))
    return load_document(path=path, url=url)


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = index
    return chunks


def ingest(recreate: bool = True, url: str | None = None) -> int:
    wait_for_qdrant()
    documents = prepare_document(url=url)
    chunks = split_documents(documents)
    collection = env("QDRANT_COLLECTION", "state_of_union_rag")
    client = make_qdrant_client()

    if recreate and client.collection_exists(collection):
        client.delete_collection(collection)

    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=make_embeddings(),
        url=env("QDRANT_URL", "http://localhost:6333"),
        collection_name=collection,
        retrieval_mode=RetrievalMode.DENSE,
    )
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare documents for the Gemma RAG demo.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download", help="Download the practice document.")
    ingest_parser = subparsers.add_parser("ingest", help="Download, split and index the document.")
    ingest_parser.add_argument("--no-recreate", action="store_true", help="Keep an existing collection.")

    args = parser.parse_args()

    if args.command == "download":
        path = download_document()
        print(f"Downloaded: {path}")
    elif args.command == "ingest":
        count = ingest(recreate=not args.no_recreate)
        print(f"Indexed chunks: {count}")


if __name__ == "__main__":
    main()
