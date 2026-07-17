# Продвинутые retriever-методы LangChain — локально

Проект воспроизводит все техники из ноутбука `Build a Smarter Search with
LangChain Context Retrieval.ipynb`. Вместо Watsonx используется локальная
Gemma через Ollama; API-ключ и облачный сервис не нужны.

## Что реализовано

- vector store backed retrieval: similarity, MMR и score threshold;
- multi-query retrieval: Gemma генерирует варианты запроса, затем выполняется union результатов;
- self-query retrieval: Gemma превращает фразу в типизированный `SearchPlan`, после чего применяются metadata-фильтры;
- parent-document retrieval: поиск по маленьким child chunks с возвратом parent context.

Все retriever-классы реализуют интерфейс LangChain `BaseRetriever` и вызываются
одинаково: `retriever.invoke("запрос")`.

## Запуск

На Ubuntu сначала установите Ollama и загрузите ту же модель, что использовалась
в предыдущих решениях:

```bash
ollama serve
ollama pull gemma4:e2b
```

В другом терминале, из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r advanced_langchain_retrievers/requirements.txt
python advanced_langchain_retrievers/demo.py
python -m unittest discover -s advanced_langchain_retrievers/tests -v
```

Можно запустить один пример:

```bash
python advanced_langchain_retrievers/demo.py vector
python advanced_langchain_retrievers/demo.py multi
python advanced_langchain_retrievers/demo.py self
python advanced_langchain_retrievers/demo.py parent
```

По умолчанию используется `gemma4:e2b`. Переопределение:

```bash
export OLLAMA_MODEL=gemma3:4b
export OLLAMA_BASE_URL=http://localhost:11434
```

Embeddings здесь намеренно учебные и локальные: слова хешируются в фиксированный
вектор, а Gemma отвечает за специализированные LLM-операции. Для production
замените `LocalHashEmbeddings` на `OllamaEmbeddings`/embedding-модель и
`LocalVectorIndex` на Chroma, FAISS, Qdrant или другой vector store — интерфейс
retriever останется тем же.
