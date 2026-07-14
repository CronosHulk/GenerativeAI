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
            "Ти корисний асистент. Розв'язуй задачу покроково і показуй коротке міркування.",
        ),
        (
            "human",
            """Розв'яжи задачу методом Chain-of-thought.

Задача:
{question}

Формат відповіді:
Міркування:
1. ...
2. ...
3. ...

Відповідь: ...""",
        ),
    ]
)

chain = prompt | llm | StrOutputParser()


if __name__ == "__main__":
    question = (
        "У Марії було 12 яблук. Вона віддала 3 яблука брату, "
        "а потім купила ще 5 яблук. Скільки яблук стало у Марії?"
    )

    answer = chain.invoke({"question": question})
    print(answer)
