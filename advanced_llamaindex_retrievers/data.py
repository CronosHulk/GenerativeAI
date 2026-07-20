from llama_index.core.schema import Document


POLICIES = [
    Document(text="Email is for business communication. Never send passwords or confidential data by email.", metadata={"section": "email"}),
    Document(text="Smoking and vaping are forbidden inside offices. Use the designated outdoor smoking area.", metadata={"section": "safety"}),
    Document(text="Employees may work remotely two days per week after manager approval.", metadata={"section": "remote work"}),
    Document(text="Company devices must use disk encryption and automatic screen locking.", metadata={"section": "security"}),
    Document(text="Annual leave requests should be submitted at least two weeks in advance.", metadata={"section": "leave"}),
]

MOVIES = [
    Document(text="Scientists bring dinosaurs back and chaos follows.", metadata={"title": "Jurassic Park", "year": 1993, "rating": 7.7, "genre": "science fiction", "director": "Steven Spielberg"}),
    Document(text="A thief enters layered dreams to plant an idea.", metadata={"title": "Inception", "year": 2010, "rating": 8.2, "genre": "science fiction", "director": "Christopher Nolan"}),
    Document(text="A detective becomes trapped in dreams that alter reality.", metadata={"title": "Paprika", "year": 2006, "rating": 8.6, "genre": "animated", "director": "Satoshi Kon"}),
    Document(text="Three men travel through the mysterious Zone.", metadata={"title": "Stalker", "year": 1979, "rating": 9.0, "genre": "science fiction", "director": "Andrei Tarkovsky"}),
    Document(text="Four sisters grow up and choose their own paths.", metadata={"title": "Little Women", "year": 2019, "rating": 8.3, "genre": "drama", "director": "Greta Gerwig"}),
]

PARENT_TEXTS = [
    Document(text="Workplace safety policy. Smoking and vaping are forbidden in every indoor company area. Employees who smoke must use the marked outdoor zone behind the west parking lot. Repeated violations are reported to HR. Fire exits must remain unobstructed at all times.", metadata={"title": "Safety handbook"}),
    Document(text="Information security policy. Every laptop must use full-disk encryption. Screens lock automatically after five minutes. Passwords must never be shared by email or chat. Suspected phishing messages should be forwarded to the security team.", metadata={"title": "Security handbook"}),
]

TECH_DOCS = [
    Document(
        text=(
            "Vector retrieval converts each document chunk into a dense embedding. "
            "It is useful when the query and the document use different words but share the same meaning."
        ),
        metadata={"doc_id": "vector", "title": "Vector Index Retriever", "topic": "semantic search"},
    ),
    Document(
        text=(
            "BM25 retrieval ranks documents with lexical signals such as term frequency, inverse document frequency, "
            "and document length normalization. It is strong for exact keywords, names, and product codes."
        ),
        metadata={"doc_id": "bm25", "title": "BM25 Retriever", "topic": "keyword search"},
    ),
    Document(
        text=(
            "Document summary retrieval generates a concise summary for each source document. "
            "Queries first select relevant summaries, then the system returns the original full document."
        ),
        metadata={"doc_id": "summary", "title": "Document Summary Index", "topic": "document selection"},
    ),
    Document(
        text=(
            "Query fusion sends multiple query variants to several retrievers and combines their rankings. "
            "Fusion modes such as reciprocal rank fusion and relative score fusion improve recall."
        ),
        metadata={"doc_id": "fusion", "title": "Query Fusion Retriever", "topic": "fusion"},
    ),
]

REFERENCE_DOCS = [
    Document(
        text="Start here for retrieval systems. For semantic search read [[vector]]. For exact matching read [[bm25]].",
        metadata={"doc_id": "overview", "title": "Retriever Overview", "refs": ["vector", "bm25"]},
    ),
    Document(
        text="Vector search stores dense embeddings and retrieves chunks by semantic similarity.",
        metadata={"doc_id": "vector", "title": "Vector Details", "refs": []},
    ),
    Document(
        text="BM25 search uses sparse lexical statistics and is effective for exact terms.",
        metadata={"doc_id": "bm25", "title": "BM25 Details", "refs": []},
    ),
]
