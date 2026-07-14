import os

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from prompts import json_parser, json_prompt, text_prompt


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def make_llm(temperature: float = 0.2) -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )


text_chain = text_prompt | make_llm(temperature=0.3) | StrOutputParser()
json_chain = json_prompt | make_llm(temperature=0.1) | json_parser


def generate_text(user_prompt: str) -> str:
    return text_chain.invoke({"user_prompt": user_prompt})


def generate_json(user_prompt: str) -> dict:
    return json_chain.invoke({"user_prompt": user_prompt})
