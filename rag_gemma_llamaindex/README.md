# RAG with Gemma, LlamaIndex and Qdrant

Такий самий RAG-сценарій, як у LangChain-папці, але data/index/query layer зроблений на LlamaIndex.

## Структура

- `prepare_documents.py` - download, LlamaIndex `Document`, `SentenceSplitter`, indexing у Qdrant.
- `rag_pipeline.py` - Qdrant-backed `VectorStoreIndex`, query engine, відповідь і sources.
- `gradio_app.py` - UI для URL, питання, відповіді і повторного питання.
- `settings.py` - Ollama, embeddings, Qdrant і env defaults.
- `docker-compose.yml` - Qdrant, якщо він ще не запущений.

## Запуск

```bash
cd rag_gemma_llamaindex
cp .env.example .env
../.venv/bin/pip install -r requirements.txt
docker compose up -d
ollama serve
```

Моделі:

```bash
ollama pull gemma4:e2b
ollama pull nomic-embed-text
```

Якщо Qdrant уже запущений з LangChain-варіанта на `localhost:6333`, другий `docker compose up -d` не потрібен.

## CLI

Індексація:

```bash
../.venv/bin/python prepare_documents.py ingest
```

Питання:

```bash
../.venv/bin/python rag_pipeline.py "What did the President say about the economy?" --sources
```

## Gradio UI

```bash
../.venv/bin/python gradio_app.py
```

UI буде доступний тут:

```text
http://localhost:7861
```

## Налаштування

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:e2b
OLLAMA_EMBED_MODEL=nomic-embed-text
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=llamaindex_state_of_union_rag
```
