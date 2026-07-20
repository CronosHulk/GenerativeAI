# RAG with Gemma, LangChain LCEL and Chroma

Мини-проект для упражнений с Retrieval-Augmented Generation:

- документ скачивается из URL IBM course;
- текст режется на чанки;
- чанки кладутся в Chroma из Docker;
- embeddings явно не создаются в коде: документы добавляются в Chroma как text documents, а Chroma использует свою default embedding function;
- вопрос проходит через LangChain LCEL pipe;
- Gemma через Ollama отвечает только по найденному контексту;
- при необходимости печатаются исходные чанки.

## Структура

- `prepare_documents.py` - скачивание, chunking и индексация документа.
- `rag_pipeline.py` - RAG chain и вопросы к уже готовому индексу.
- `gradio_app.py` - Gradio-интерфейс для загрузки документа и вопросов.
- `settings.py` - общие настройки, Ollama/Chroma клиенты.
- `docker-compose.yml` - Chroma DB.
- `requirements.txt` - зависимости проекта.
- `.env.example` - пример настроек моделей и URL.
- `data/stateOfUnion.txt` - появится после скачивания документа.

## Запуск

Из корня репозитория:

```bash
cd rag_gemma_langchain_chroma
cp .env.example .env
../.venv/bin/pip install -r requirements.txt
docker compose up -d
ollama serve
```

В отдельном терминале подтяните Gemma:

```bash
ollama pull gemma4:e2b
```

Если у вас другая Gemma-модель в Ollama, поменяйте `.env`:

```bash
OLLAMA_MODEL=gemma3:4b
```

## Индексация документа

```bash
../.venv/bin/python prepare_documents.py ingest
```

Команда сама скачает документ:

```text
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/XVnuuEg94sAE4S_xAsGxBA.txt
```

## Вопрос к RAG

```bash
../.venv/bin/python rag_pipeline.py "What did the President say about the economy?" --sources
```

Еще пример:

```bash
../.venv/bin/python rag_pipeline.py "Summarize the main points of the document in 5 bullets."
```

## Gradio UI

```bash
../.venv/bin/python gradio_app.py
```

Интерфейс откроется на:

```text
http://localhost:7860
```

В UI есть поле URL с дефолтом из `.env`, кнопка загрузки/индексации, поле вопроса, экран ответа с sources и кнопка возврата к следующему вопросу.

## Где здесь LangChain pipes

Основная цепочка находится в `build_rag_chain()`:

```python
with_context = RunnableParallel(
    {
        "question": RunnablePassthrough(),
        "source_documents": retriever,
    }
) | RunnablePassthrough.assign(
    context=RunnableLambda(lambda values: format_docs(values["source_documents"]))
)

chain = with_context | {
    "answer": prompt | make_llm() | StrOutputParser(),
    "source_documents": itemgetter("source_documents"),
}
```

Retrieval делает Chroma через `collection.query(query_texts=[question])`, то есть query embedding тоже считается Chroma default embedding function.

## Полезные настройки

Все настройки можно менять через `.env`:

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:e2b
CHROMA_URL=http://localhost:8000
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_COLLECTION=state_of_union_rag
```

Для остановки Chroma:

```bash
docker compose down
```
