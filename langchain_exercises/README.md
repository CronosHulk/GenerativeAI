# LangChain Exercises with Gemma

These exercises use LangChain and a local Gemma model through Ollama.

Before running:

```bash
ollama serve
ollama pull gemma4:e2b
```

You can override the model name:

```bash
export OLLAMA_MODEL="gemma4:e2b"
export OLLAMA_BASE_URL="http://localhost:11434"
```

Files:

- `exercise_01_model_parameters.py` compares low and high temperature responses.
- `exercise_02_json_output_parser.py` parses model output into a Python dictionary.
- `exercise_03_document_loaders_splitters.py` loads documents and compares chunking settings.
- `exercise_04_simple_retrieval_system.py` builds a small RAG workflow.
- `exercise_05_chatbot_with_memory.py` builds a chatbot with conversation history.
- `exercise_06_multi_step_chains.py` compares sequential and LCEL chain approaches.
- `exercise_07_basic_agent_tools.py` creates an agent with calculator and text formatter tools.

Run one exercise:

```bash
../.venv/bin/python exercise_02_json_output_parser.py
```
