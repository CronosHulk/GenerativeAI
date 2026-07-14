from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="gemma4:e2b",
    base_url="http://localhost:11434",
    temperature=0.2,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты полезный ассистент. Решай задачу пошагово и показывай краткое рассуждение.",
        ),
        (
            "human",
            """Реши задачу методом Chain-of-thought.

Задача:
{question}

Формат ответа:
Рассуждение:
1. ...
2. ...
3. ...

Ответ: ...""",
        ),
    ]
)

chain = prompt | llm | StrOutputParser()


if __name__ == "__main__":
    question = (
        "У Маши было 12 яблок. Она отдала 3 яблока брату, "
        "а потом купила еще 5 яблок. Сколько яблок стало у Маши?"
    )

    answer = chain.invoke({"question": question})
    print(answer)
