# Flask GenAI App з Gemma

## Завдання

Зібрати невеликий backend для GenAI application:

- приймати HTTP-запит з prompt;
- викликати локальну LLM Gemma через Ollama та LangChain;
- повертати звичайну текстову відповідь;
- повертати структуровану JSON-відповідь через `JsonOutputParser`.

Watsonx, IBM models і API keys тут не використовуються.

## Структура

- `app.py` — Flask routes.
- `model.py` — створення Gemma LLM і LangChain chains.
- `prompts.py` — prompt templates і JSON parser.
- `llm_test.py` — швидкий тест без HTTP.
- `requirements.txt` — залежності саме для Flask-версії.

## Запуск

З кореня проєкту:

```bash
cd flask_genai_app
../.venv/bin/pip install -r requirements.txt
ollama serve
ollama pull gemma4:e2b
../.venv/bin/python app.py
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

Health check:

```bash
curl http://localhost:5000/health
```

Text response:

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is LangChain in simple terms?"}'
```

Structured JSON response:

```bash
curl -X POST http://localhost:5000/generate-json \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Give me 3 tips for learning Python."}'
```
