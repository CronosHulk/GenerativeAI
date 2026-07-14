from pathlib import Path
from urllib.request import urlopen

from langchain_core.documents import Document

from common import print_section


DATA_DIR = Path(__file__).parent / "data"


def load_text_file(path: Path) -> list[Document]:
    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": str(path), "loader": "local_text"},
        )
    ]


def load_web_page(url: str) -> list[Document]:
    html = urlopen(url, timeout=10).read().decode("utf-8", errors="ignore")
    text = " ".join(html.split())
    return [Document(page_content=text, metadata={"source": url, "loader": "web"})]


def load_pdf(path: Path) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install pypdf to load PDF files: pip install pypdf") from exc

    reader = PdfReader(str(path))
    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        documents.append(
            Document(
                page_content=page.extract_text() or "",
                metadata={
                    "source": str(path),
                    "loader": "pdf",
                    "page": page_number,
                },
            )
        )
    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    chunks = []

    for document in documents:
        text = document.page_content
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                metadata = {
                    **document.metadata,
                    "chunk_index": chunk_index,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                }
                chunks.append(Document(page_content=chunk_text, metadata=metadata))

            chunk_index += 1
            start = max(end - chunk_overlap, start + 1)

    return chunks


def display_chunk_stats(name: str, chunks: list[Document]) -> None:
    lengths = [len(chunk.page_content) for chunk in chunks]
    print_section(name)
    print(f"Chunks: {len(chunks)}")
    print(f"Shortest chunk: {min(lengths)} characters")
    print(f"Longest chunk: {max(lengths)} characters")
    print(f"Average length: {sum(lengths) / len(lengths):.1f} characters")
    print(f"First metadata: {chunks[0].metadata}")
    print(f"First preview: {chunks[0].page_content[:220]}...")


if __name__ == "__main__":
    docs = load_text_file(DATA_DIR / "langchain_architecture.txt")

    small_chunks = split_documents(docs, chunk_size=350, chunk_overlap=50)
    large_chunks = split_documents(docs, chunk_size=700, chunk_overlap=100)

    display_chunk_stats("Small splitter", small_chunks)
    display_chunk_stats("Large splitter", large_chunks)

    print_section("Metadata preservation")
    for chunk in small_chunks[:3]:
        print(chunk.metadata)
