from __future__ import annotations

import argparse

from data import load_20_newsgroups
from embeddings import HashSentenceEncoder, UniversalSentenceEncoder
from search_engine import SemanticSearchEngine


def make_encoder(name: str):
    if name == "use":
        return UniversalSentenceEncoder()
    return HashSentenceEncoder()


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic similarity search with FAISS-style indexing.")
    parser.add_argument("query", nargs="?", default="motorcycle", help="Search query.")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest documents to return.")
    parser.add_argument("--limit", type=int, default=500, help="Dataset document limit.")
    parser.add_argument("--encoder", choices=["hash", "use"], default="hash", help="Embedding backend.")
    parser.add_argument("--numpy-index", action="store_true", help="Use the NumPy fallback instead of FAISS.")
    args = parser.parse_args()

    documents, target_names = load_20_newsgroups(limit=args.limit)
    print(f"Loaded {len(documents)} documents. Topics available: {len(target_names)}")

    engine = SemanticSearchEngine(encoder=make_encoder(args.encoder), prefer_faiss=not args.numpy_index)
    engine.build(documents)

    print(f"\nQuery: {args.query}\n")
    for result in engine.search(args.query, k=args.k):
        preview = " ".join(result.processed_text.split())[:600]
        print(f"Rank {result.rank}: distance={result.distance:.4f} index={result.index}")
        print(preview)
        print("-" * 80)


if __name__ == "__main__":
    main()
