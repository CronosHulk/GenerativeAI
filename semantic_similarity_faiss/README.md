# Semantic Similarity with FAISS

Локальная реализация lab `Semantic Similarity with FAISS`.

## Что внутри

- preprocessing для 20 Newsgroups-style текста;
- optional loader для `fetch_20newsgroups`;
- Universal Sentence Encoder wrapper через TensorFlow Hub;
- lightweight `HashSentenceEncoder` для запуска без TensorFlow;
- FAISS `IndexFlatL2` с NumPy fallback;
- CLI semantic search demo;
- unittest-тесты.

## Быстрый запуск без тяжелых зависимостей

```bash
PYTHONDONTWRITEBYTECODE=1 python semantic_similarity_faiss/demo.py "motorcycle helmet tires" --numpy-index
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s semantic_similarity_faiss/tests -v
```

Если `scikit-learn` не установлен, demo использует маленький локальный sample
dataset. Если `faiss-cpu` не установлен, используется NumPy fallback.

## Запуск как в lab

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r semantic_similarity_faiss/requirements.txt
python semantic_similarity_faiss/demo.py "motorcycle" --encoder use
```

`--encoder use` загружает Universal Sentence Encoder:

```text
https://tfhub.dev/google/universal-sentence-encoder/4
```

Для больших коллекций используйте FAISS напрямую. Для учебного режима можно
оставить `--numpy-index`, чтобы видеть тот же `add/search` flow без внешней
библиотеки.
