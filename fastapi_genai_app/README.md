# FastAPI GenAI App з Gemma

## Завдання

Зібрати невеликий backend для GenAI application:

- приймати HTTP-запит з prompt;
- викликати локальну LLM Gemma через Ollama та LangChain;
- повертати звичайну текстову відповідь;
- повертати структуровану JSON-відповідь через `JsonOutputParser`;
- мати автоматичну OpenAPI-документацію від FastAPI.

Watsonx, IBM models і API keys тут не використовуються.

## Структура

- `app.py` — FastAPI routes і Pydantic schemas.
- `model.py` — створення Gemma LLM і LangChain chains.
- `prompts.py` — prompt templates і JSON parser.
- `llm_test.py` — швидкий тест без HTTP.
- `requirements.txt` — залежності саме для FastAPI-версії.

## Запуск

З кореня проєкту:

```bash
cd fastapi_genai_app
../.venv/bin/pip install -r requirements.txt
ollama serve
ollama pull gemma4:e2b
../.venv/bin/uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Якщо модель має іншу назву:

```bash
export OLLAMA_MODEL="gemma4:e2b"
export OLLAMA_BASE_URL="http://localhost:11434"
```

## Перевірка LLM без сервера

```bash
../.venv/bin/python llm_test.py
```

## HTTP endpoints

OpenAPI docs:

```text
http://localhost:8000/docs
```

Health check:

```bash
curl http://localhost:8000/health
```

Text response:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is LangChain in simple terms?"}'
```

Structured JSON response:

```bash
curl -X POST http://localhost:8000/generate-json \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Give me 3 tips for learning Python."}'
```
