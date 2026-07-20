from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import Document, NodeWithScore, QueryBundle, TextNode


TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)


def node_text(node: Document | TextNode) -> str:
    return node.get_content(metadata_mode="none")


class LocalHashEmbedding:
    """Small deterministic embeddings for offline LlamaIndex demonstrations."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            slot = int.from_bytes(digest, "big") % self.dimensions
            vector[slot] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def as_node_with_score(document: Document | TextNode, score: float | None = None) -> NodeWithScore:
    return NodeWithScore(node=document, score=score)


def dedupe_key(node: Document | TextNode) -> tuple[str, tuple]:
    return node_text(node), tuple(sorted(node.metadata.items()))


class LocalVectorIndex:
    def __init__(self, documents: Iterable[Document | TextNode], embeddings: LocalHashEmbedding | None = None):
        self.documents = list(documents)
        self.embeddings = embeddings or LocalHashEmbedding()
        self.vectors = self.embeddings.embed_documents([node_text(doc) for doc in self.documents])

    def scored(self, query: str) -> list[tuple[Document | TextNode, float]]:
        query_vector = self.embeddings.embed(query)
        pairs = [(doc, cosine(query_vector, vector)) for doc, vector in zip(self.documents, self.vectors)]
        return sorted(pairs, key=lambda item: item[1], reverse=True)

    def similarity(self, query: str, k: int = 4, threshold: float | None = None) -> list[NodeWithScore]:
        pairs = self.scored(query)
        if threshold is not None:
            pairs = [pair for pair in pairs if pair[1] >= threshold]
        return [as_node_with_score(doc, score) for doc, score in pairs[:k]]

    def mmr(self, query: str, k: int = 4, fetch_k: int = 10, lambda_mult: float = 0.5) -> list[NodeWithScore]:
        candidates = self.scored(query)[:fetch_k]
        selected: list[tuple[Document | TextNode, float, list[float]]] = []
        while candidates and len(selected) < k:
            best_position, best_score = 0, float("-inf")
            for position, (doc, relevance) in enumerate(candidates):
                vector = self.embeddings.embed(node_text(doc))
                redundancy = max((cosine(vector, chosen_vector) for _, _, chosen_vector in selected), default=0.0)
                score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
                if score > best_score:
                    best_position, best_score = position, score
            doc, relevance = candidates.pop(best_position)
            selected.append((doc, relevance, self.embeddings.embed(node_text(doc))))
        return [as_node_with_score(doc, relevance) for doc, relevance, _ in selected]


class VectorStoreRetriever(BaseRetriever):
    def __init__(
        self,
        index: LocalVectorIndex,
        search_type: str = "similarity",
        k: int = 4,
        threshold: float | None = None,
    ):
        super().__init__()
        self.index = index
        self.search_type = search_type
        self.k = k
        self.threshold = threshold

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query = query_bundle.query_str
        if self.search_type == "mmr":
            return self.index.mmr(query, k=self.k)
        if self.search_type == "similarity_score_threshold":
            return self.index.similarity(query, k=self.k, threshold=self.threshold)
        return self.index.similarity(query, k=self.k)


class MultiQueryRetriever(BaseRetriever):
    def __init__(self, base_retriever: BaseRetriever, query_generator: Callable[[str], list[str]]):
        super().__init__()
        self.base_retriever = base_retriever
        self.query_generator = query_generator

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        unique: dict[tuple[str, tuple], NodeWithScore] = {}
        query = query_bundle.query_str
        for variant in [query, *self.query_generator(query)]:
            for node_with_score in self.base_retriever.retrieve(variant):
                node = node_with_score.node
                unique[dedupe_key(node)] = node_with_score
        return list(unique.values())


def local_query_variants(query: str) -> list[str]:
    """Rule-based stand-in for the LLM normally used by multi-query retrieval."""
    lowered = query.lower()
    variants = [f"rules and requirements about {query}", f"company handbook {query}"]
    synonyms = {"smoking": "vaping tobacco outdoor area", "email": "messages communication confidential data", "remote": "work from home manager approval"}
    variants.extend(value for key, value in synonyms.items() if key in lowered)
    return variants


class SelfQueryRetriever(BaseRetriever):
    def __init__(self, index: LocalVectorIndex, k: int = 4, planner: Callable[[str], Any] | None = None):
        super().__init__()
        self.index = index
        self.k = k
        self.planner = planner

    def _parse(self, query: str) -> tuple[str, list[Callable[[dict], bool]]]:
        filters: list[Callable[[dict], bool]] = []
        lowered = query.lower()
        rating = re.search(r"(?:rating|rated|рейтинг\w*)\s*(?:higher than|above|over|выше)?\s*(\d+(?:\.\d+)?)", lowered)
        if rating:
            value = float(rating.group(1))
            filters.append(lambda meta, value=value: float(meta.get("rating", 0)) > value)
        year = re.search(r"(?:after|после)\s*(\d{4})", lowered)
        if year:
            value = int(year.group(1))
            filters.append(lambda meta, value=value: int(meta.get("year", 0)) > value)
        for field in ("director", "genre"):
            values = {str(doc.metadata.get(field, "")) for doc in self.index.documents}
            for value in values:
                if value and value.lower() in lowered:
                    filters.append(lambda meta, field=field, value=value: meta.get(field) == value)
        semantic = re.sub(r"(?:rating|rated|рейтинг\w*)[^,?.]*|(?:after|после)\s*\d{4}", "", query, flags=re.I).strip(" ,?.")
        return semantic, filters

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query = query_bundle.query_str
        if self.planner is None:
            semantic, filters = self._parse(query)
        else:
            plan = self.planner(query)
            semantic = plan.semantic_query
            filters = []
            if plan.min_rating is not None:
                filters.append(lambda meta, value=plan.min_rating: float(meta.get("rating", 0)) > value)
            if plan.after_year is not None:
                filters.append(lambda meta, value=plan.after_year: int(meta.get("year", 0)) > value)
            if plan.director:
                filters.append(lambda meta, value=plan.director.lower(): str(meta.get("director", "")).lower() == value)
            if plan.genre:
                filters.append(lambda meta, value=plan.genre.lower(): str(meta.get("genre", "")).lower() == value)
        candidates = self.index.scored(semantic or query)
        return [as_node_with_score(doc, score) for doc, score in candidates if all(rule(doc.metadata) for rule in filters)][: self.k]


def split_words(text: str, size: int = 12, overlap: int = 3) -> list[str]:
    words = text.split()
    step = max(1, size - overlap)
    return [" ".join(words[start:start + size]) for start in range(0, len(words), step) if words[start:start + size]]


class ParentDocumentRetriever(BaseRetriever):
    def __init__(self, child_index: LocalVectorIndex, parents: dict[str, Document], k_children: int = 4):
        super().__init__()
        self.child_index = child_index
        self.parents = parents
        self.k_children = k_children

    @classmethod
    def from_documents(cls, documents: Iterable[Document], child_size: int = 12, overlap: int = 3):
        parents: dict[str, Document] = {}
        children: list[TextNode] = []
        for number, parent in enumerate(documents):
            parent_id = f"parent-{number}"
            parents[parent_id] = parent
            for chunk in split_words(node_text(parent), child_size, overlap):
                children.append(TextNode(text=chunk, metadata={"parent_id": parent_id}))
        return cls(child_index=LocalVectorIndex(children), parents=parents)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        result: list[NodeWithScore] = []
        seen: set[str] = set()
        for child in self.child_index.similarity(query_bundle.query_str, k=self.k_children):
            parent_id = child.node.metadata["parent_id"]
            if parent_id not in seen:
                result.append(as_node_with_score(self.parents[parent_id], child.score))
                seen.add(parent_id)
        return result


class BM25Retriever(BaseRetriever):
    """Local BM25 retriever that mirrors LlamaIndex's keyword-search concept."""

    def __init__(self, documents: Iterable[Document | TextNode], k: int = 4, k1: float = 1.5, b: float = 0.75):
        super().__init__()
        self.documents = list(documents)
        self.k = k
        self.k1 = k1
        self.b = b
        self.tokenized = [TOKEN_RE.findall(node_text(doc).lower()) for doc in self.documents]
        self.term_counts = [Counter(tokens) for tokens in self.tokenized]
        self.doc_lengths = [len(tokens) for tokens in self.tokenized]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        doc_freq: Counter[str] = Counter()
        for tokens in self.tokenized:
            doc_freq.update(set(tokens))
        total = len(self.documents) or 1
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in doc_freq.items()
        }

    def _score(self, query: str, position: int) -> float:
        score = 0.0
        length = self.doc_lengths[position] or 1
        counts = self.term_counts[position]
        for term in TOKEN_RE.findall(query.lower()):
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + self.k1 * (1 - self.b + self.b * length / (self.avgdl or 1))
            score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
        return score

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        scored = [
            (doc, self._score(query_bundle.query_str, position))
            for position, doc in enumerate(self.documents)
        ]
        scored = [pair for pair in scored if pair[1] > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [as_node_with_score(doc, score) for doc, score in scored[: self.k]]


def default_summary(text: str, max_words: int = 24) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    words = sentence.split()
    return " ".join(words[:max_words])


class DocumentSummaryRetriever(BaseRetriever):
    """Selects documents through generated summaries, then returns full documents."""

    def __init__(
        self,
        documents: Iterable[Document],
        k: int = 2,
        summarizer: Callable[[Document], str] | None = None,
    ):
        super().__init__()
        self.documents: dict[str, Document] = {}
        summary_nodes: list[TextNode] = []
        summarizer = summarizer or (lambda doc: default_summary(node_text(doc)))
        for number, document in enumerate(documents):
            doc_id = str(document.metadata.get("doc_id", f"doc-{number}"))
            self.documents[doc_id] = document
            summary_nodes.append(
                TextNode(
                    text=summarizer(document),
                    metadata={
                        "doc_id": doc_id,
                        "title": document.metadata.get("title", doc_id),
                        "kind": "summary",
                    },
                )
            )
        self.summary_index = LocalVectorIndex(summary_nodes)
        self.k = k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        result: list[NodeWithScore] = []
        seen: set[str] = set()
        for summary in self.summary_index.similarity(query_bundle.query_str, k=self.k):
            doc_id = summary.node.metadata["doc_id"]
            if doc_id not in seen:
                result.append(as_node_with_score(self.documents[doc_id], summary.score))
                seen.add(doc_id)
        return result


class AutoMergingRetriever(BaseRetriever):
    """Returns a parent node when enough matching child chunks point to it."""

    def __init__(
        self,
        child_index: LocalVectorIndex,
        parents: dict[str, Document],
        child_counts: dict[str, int],
        k_children: int = 6,
        merge_threshold: float = 0.5,
    ):
        super().__init__()
        self.child_index = child_index
        self.parents = parents
        self.child_counts = child_counts
        self.k_children = k_children
        self.merge_threshold = merge_threshold

    @classmethod
    def from_documents(cls, documents: Iterable[Document], child_size: int = 10, overlap: int = 2):
        parents: dict[str, Document] = {}
        child_counts: dict[str, int] = {}
        children: list[TextNode] = []
        for number, parent in enumerate(documents):
            parent_id = str(parent.metadata.get("doc_id", f"parent-{number}"))
            parents[parent_id] = parent
            chunks = split_words(node_text(parent), child_size, overlap)
            child_counts[parent_id] = len(chunks)
            for chunk_number, chunk in enumerate(chunks):
                children.append(TextNode(text=chunk, metadata={"parent_id": parent_id, "chunk": chunk_number}))
        return cls(LocalVectorIndex(children), parents, child_counts)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        children = self.child_index.similarity(query_bundle.query_str, k=self.k_children)
        grouped: dict[str, list[NodeWithScore]] = defaultdict(list)
        for child in children:
            grouped[child.node.metadata["parent_id"]].append(child)

        result: list[NodeWithScore] = []
        for parent_id, matches in grouped.items():
            coverage = len(matches) / max(1, self.child_counts[parent_id])
            best_score = max(match.score or 0.0 for match in matches)
            if len(matches) >= 2 and coverage >= self.merge_threshold:
                result.append(as_node_with_score(self.parents[parent_id], best_score))
            else:
                result.extend(matches)
        return sorted(result, key=lambda item: item.score or 0.0, reverse=True)


class RecursiveRetriever(BaseRetriever):
    """Follows references from retrieved routing nodes to target documents."""

    def __init__(self, entry_retriever: BaseRetriever, references: dict[str, Document], max_depth: int = 2):
        super().__init__()
        self.entry_retriever = entry_retriever
        self.references = references
        self.max_depth = max_depth

    def _follow(self, node_with_score: NodeWithScore, depth: int, result: list[NodeWithScore], seen: set[str]) -> None:
        node = node_with_score.node
        doc_id = str(node.metadata.get("doc_id", dedupe_key(node)))
        if doc_id not in seen:
            result.append(node_with_score)
            seen.add(doc_id)
        if depth >= self.max_depth:
            return
        for ref_id in node.metadata.get("refs", []):
            target = self.references.get(ref_id)
            if target is not None:
                self._follow(as_node_with_score(target, node_with_score.score), depth + 1, result, seen)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        result: list[NodeWithScore] = []
        seen: set[str] = set()
        for node_with_score in self.entry_retriever.retrieve(query_bundle.query_str):
            self._follow(node_with_score, depth=0, result=result, seen=seen)
        return result


class QueryFusionRetriever(BaseRetriever):
    """Runs query variants across retrievers and fuses rankings."""

    def __init__(
        self,
        retrievers: Sequence[BaseRetriever],
        query_generator: Callable[[str], list[str]] = local_query_variants,
        k: int = 4,
        mode: str = "rrf",
        rrf_k: int = 60,
    ):
        super().__init__()
        self.retrievers = list(retrievers)
        self.query_generator = query_generator
        self.k = k
        self.mode = mode
        self.rrf_k = rrf_k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        rankings = [
            retriever.retrieve(query)
            for query in [query_bundle.query_str, *self.query_generator(query_bundle.query_str)]
            for retriever in self.retrievers
        ]
        if self.mode == "relative_score":
            return self._relative_score(rankings)
        if self.mode == "distribution":
            return self._distribution_score(rankings)
        return self._rrf(rankings)

    def _rrf(self, rankings: list[list[NodeWithScore]]) -> list[NodeWithScore]:
        fused: dict[tuple[str, tuple], tuple[NodeWithScore, float]] = {}
        for ranking in rankings:
            for rank, item in enumerate(ranking, start=1):
                key = dedupe_key(item.node)
                current, score = fused.get(key, (item, 0.0))
                fused[key] = (current, score + 1 / (self.rrf_k + rank))
        return self._top(fused)

    def _relative_score(self, rankings: list[list[NodeWithScore]]) -> list[NodeWithScore]:
        fused: dict[tuple[str, tuple], tuple[NodeWithScore, float]] = {}
        for ranking in rankings:
            scores = [item.score or 0.0 for item in ranking]
            low, high = (min(scores), max(scores)) if scores else (0.0, 0.0)
            span = high - low or 1.0
            for item in ranking:
                key = dedupe_key(item.node)
                current, score = fused.get(key, (item, 0.0))
                fused[key] = (current, score + ((item.score or 0.0) - low) / span)
        return self._top(fused)

    def _distribution_score(self, rankings: list[list[NodeWithScore]]) -> list[NodeWithScore]:
        fused: dict[tuple[str, tuple], tuple[NodeWithScore, float]] = {}
        for ranking in rankings:
            scores = [item.score or 0.0 for item in ranking]
            mean = sum(scores) / len(scores) if scores else 0.0
            variance = sum((score - mean) ** 2 for score in scores) / len(scores) if scores else 0.0
            stddev = math.sqrt(variance) or 1.0
            for item in ranking:
                key = dedupe_key(item.node)
                current, score = fused.get(key, (item, 0.0))
                fused[key] = (current, score + ((item.score or 0.0) - mean) / stddev)
        return self._top(fused)

    def _top(self, fused: dict[tuple[str, tuple], tuple[NodeWithScore, float]]) -> list[NodeWithScore]:
        nodes = [as_node_with_score(item.node, score) for item, score in fused.values()]
        nodes.sort(key=lambda item: item.score or 0.0, reverse=True)
        return nodes[: self.k]


class HybridRetriever(QueryFusionRetriever):
    def __init__(self, vector_retriever: BaseRetriever, keyword_retriever: BaseRetriever, k: int = 4):
        super().__init__(
            retrievers=[vector_retriever, keyword_retriever],
            query_generator=lambda query: [],
            k=k,
            mode="relative_score",
        )
