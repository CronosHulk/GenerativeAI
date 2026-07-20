import os
import time
from pathlib import Path
from typing import Any

import chromadb
import requests
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_ollama import ChatOllama


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_SOURCE_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "XVnuuEg94sAE4S_xAsGxBA.txt"
)


def env(name: str, default: str) -> str:
    return os.getenv(name, default)


def make_llm() -> ChatOllama:
    return ChatOllama(
        model=env("OLLAMA_MODEL", "gemma4:e2b"),
        base_url=env("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.1,
    )


def make_chroma_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(
        host=env("CHROMA_HOST", "localhost"),
        port=int(env("CHROMA_PORT", "8000")),
    )


def make_chroma_collection() -> Any:
    client = make_chroma_client()
    return client.get_or_create_collection(
        name=env("CHROMA_COLLECTION", "state_of_union_rag"),
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def wait_for_chroma(timeout_seconds: int = 30) -> None:
    url = env("CHROMA_URL", "http://localhost:8000").rstrip("/")
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            for path in ("/api/v2/heartbeat", "/api/v1/heartbeat"):
                response = requests.get(f"{url}{path}", timeout=2)
                if response.ok:
                    return
            response.raise_for_status()
        except requests.RequestException as error:
            last_error = error
            time.sleep(1)

    raise RuntimeError(
        f"Chroma is not ready at {url}. Start it with `docker compose up -d`."
    ) from last_error
