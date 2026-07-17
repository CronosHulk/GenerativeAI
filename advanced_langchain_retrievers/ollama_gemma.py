from __future__ import annotations

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


def make_gemma(temperature: float = 0.1) -> ChatOllama:
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=temperature,
    )


def make_multi_query_generator(llm: ChatOllama):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate three alternative search queries that express different perspectives on the user's question. Return one query per line, without numbering or comments."),
        ("human", "Question: {question}"),
    ])
    chain = prompt | llm | StrOutputParser()

    def generate(question: str) -> list[str]:
        lines = chain.invoke({"question": question}).splitlines()
        return [line.strip(" -0123456789.\t") for line in lines if line.strip()][:3]

    return generate


class SearchPlan(BaseModel):
    semantic_query: str = Field(description="Only the semantic movie description; omit metadata constraints")
    min_rating: float | None = Field(default=None, description="Exclusive lower rating bound")
    after_year: int | None = Field(default=None, description="Exclusive lower release-year bound")
    director: str | None = None
    genre: str | None = None


def make_self_query_planner(llm: ChatOllama):
    structured_llm = llm.with_structured_output(SearchPlan)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Convert a natural-language movie request into a semantic query and metadata constraints. Do not invent a constraint that is absent. Known genres include science fiction, drama and animated."),
        ("human", "Request: {question}"),
    ])
    chain = prompt | structured_llm

    def plan(question: str) -> SearchPlan:
        return chain.invoke({"question": question})

    return plan
