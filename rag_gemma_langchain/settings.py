import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from qdrant_client import QdrantClient


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_SOURCE_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "XVnuuEg94sAE4S_xAsGxBA.txt"
)


def env(name: str, default: str) -> str:
    return os.getenv(name, default)


def make_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=env("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        base_url=env("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def make_llm() -> ChatOllama:
    return ChatOllama(
        model=env("OLLAMA_MODEL", "gemma4:e2b"),
        base_url=env("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.1,
    )


def make_qdrant_client() -> QdrantClient:
    return QdrantClient(url=env("QDRANT_URL", "http://localhost:6333"))


def wait_for_qdrant(timeout_seconds: int = 30) -> None:
    url = env("QDRANT_URL", "http://localhost:6333").rstrip("/")
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{url}/collections", timeout=2)
            response.raise_for_status()
            return
        except requests.RequestException as error:
            last_error = error
            time.sleep(1)

    raise RuntimeError(
        f"Qdrant is not ready at {url}. Start it with `docker compose up -d`."
    ) from last_error
