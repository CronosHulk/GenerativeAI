# Продвинутые retriever-методы LlamaIndex — локально

Это локальная версия lab `Explore Advanced Retrievers in LlamaIndex`. Вместо
IBM watsonx.ai используется Gemma через Ollama, а основные retriever-примеры
работают офлайн на deterministic embeddings и локальном BM25.

## Что реализовано

- Vector Index Retriever: semantic search, MMR и score threshold;
- BM25 Retriever: keyword-based ranking по lexical signals;
- Document Summary Retriever: поиск по summary с возвратом полного документа;
- Auto Merging Retriever: child chunks автоматически сливаются в parent context;
- Recursive Retriever: найденный node может вести к связанным документам;
- Query Fusion Retriever: multi-query + fusion modes `rrf`, `relative_score`, `distribution`;
- Custom Hybrid Retriever: vector + BM25;
- Production RAG Pipeline: минимальный production-style retrieval слой с citations;
- Multi-query и self-query demos с Gemma через Ollama.

Все retriever-классы наследуются от LlamaIndex `BaseRetriever` и вызываются через
`retriever.retrieve("запрос")`. Результат — список `NodeWithScore`, как принято в
LlamaIndex.

## Запуск

На Ubuntu сначала установите Ollama и загрузите модель:

```bash
ollama serve
ollama pull gemma4:e2b
```

В другом терминале, из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r advanced_llamaindex_retrievers/requirements.txt
python advanced_llamaindex_retrievers/demo.py
python -m unittest discover -s advanced_llamaindex_retrievers/tests -v
```

Можно запустить один пример:

```bash
python advanced_llamaindex_retrievers/demo.py vector
python advanced_llamaindex_retrievers/demo.py bm25
python advanced_llamaindex_retrievers/demo.py summary
python advanced_llamaindex_retrievers/demo.py auto_merge
python advanced_llamaindex_retrievers/demo.py recursive
python advanced_llamaindex_retrievers/demo.py fusion
python advanced_llamaindex_retrievers/demo.py hybrid
python advanced_llamaindex_retrievers/demo.py production
python advanced_llamaindex_retrievers/demo.py multi
python advanced_llamaindex_retrievers/demo.py self
python advanced_llamaindex_retrievers/demo.py parent
```

По умолчанию используется `gemma4:e2b`. Переопределение:

```bash
export OLLAMA_MODEL=gemma3:4b
export OLLAMA_BASE_URL=http://localhost:11434
```

## Как это связано с БД

Эта папка специально показывает, что retrieval-фишки требуют подготовки данных:

- vector search хранит embeddings;
- BM25/keyword search строит lexical index;
- summary search хранит summaries и `doc_id`;
- auto-merge/parent retrieval хранит child chunks и `parent_id`;
- recursive retrieval хранит references между nodes/documents;
- hybrid/fusion retrieval объединяет несколько уже построенных retrievers.

Для production замените `LocalHashEmbedding`/`LocalVectorIndex` на настоящий
`VectorStoreIndex` с `OllamaEmbedding`, Qdrant, Chroma, FAISS или другим vector
store. BM25 можно оставить отдельным retriever или заменить на full-text/sparse
возможности вашей БД.
