import argparse
from pathlib import Path

import requests
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore

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
            text=text,
            metadata={
                "source": str(path),
                "url": url or env("RAG_SOURCE_URL", DEFAULT_SOURCE_URL),
            },
        )
    ]


def prepare_document(url: str | None = None) -> list[Document]:
    path = download_document(url=url or env("RAG_SOURCE_URL", DEFAULT_SOURCE_URL))
    return load_document(path=path, url=url)


def ingest(recreate: bool = True, url: str | None = None) -> int:
    wait_for_qdrant()
    collection = env("QDRANT_COLLECTION", "llamaindex_state_of_union_rag")
    client = make_qdrant_client()

    if recreate and client.collection_exists(collection):
        client.delete_collection(collection)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    documents = prepare_document(url=url)
    splitter = SentenceSplitter(chunk_size=900, chunk_overlap=150)
    nodes = splitter.get_nodes_from_documents(documents)
    for index, node in enumerate(nodes):
        node.metadata["chunk"] = index

    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=make_embeddings(),
    )
    return len(nodes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare documents for the LlamaIndex RAG demo.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download", help="Download the practice document.")
    ingest_parser = subparsers.add_parser("ingest", help="Download, split and index the document.")
    ingest_parser.add_argument("--url", default=None, help="Document URL.")
    ingest_parser.add_argument("--no-recreate", action="store_true", help="Keep an existing collection.")

    args = parser.parse_args()

    if args.command == "download":
        path = download_document()
        print(f"Downloaded: {path}")
    elif args.command == "ingest":
        count = ingest(recreate=not args.no_recreate, url=args.url)
        print(f"Indexed chunks: {count}")


if __name__ == "__main__":
    main()
