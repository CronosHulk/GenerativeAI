from __future__ import annotations

import json
import os

from llama_index.llms.ollama import Ollama
from pydantic import BaseModel, Field


def make_gemma(temperature: float = 0.1) -> Ollama:
    return Ollama(
        model=os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=temperature,
        request_timeout=120,
    )


def make_multi_query_generator(llm: Ollama):
    def generate(question: str) -> list[str]:
        prompt = (
            "Generate three alternative search queries that express different perspectives "
            "on the user's question. Return one query per line, without numbering or comments.\n\n"
            f"Question: {question}"
        )
        lines = str(llm.complete(prompt)).splitlines()
        return [line.strip(" -0123456789.\t") for line in lines if line.strip()][:3]

    return generate


class SearchPlan(BaseModel):
    semantic_query: str = Field(description="Only the semantic movie description; omit metadata constraints")
    min_rating: float | None = Field(default=None, description="Exclusive lower rating bound")
    after_year: int | None = Field(default=None, description="Exclusive lower release-year bound")
    director: str | None = None
    genre: str | None = None


def make_self_query_planner(llm: Ollama):
    def plan(question: str) -> SearchPlan:
        prompt = (
            "Convert a natural-language movie request into JSON with these keys: "
            "semantic_query, min_rating, after_year, director, genre. "
            "Use null for absent constraints. Known genres include science fiction, drama and animated. "
            "Return JSON only.\n\n"
            f"Request: {question}"
        )
        raw = str(llm.complete(prompt)).strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.removeprefix("json").strip()
        return SearchPlan.model_validate(json.loads(raw))

    return plan
