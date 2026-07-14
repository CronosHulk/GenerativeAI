from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="gemma4:e2b",
    base_url="http://localhost:11434",
    temperature=0.7,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Ти корисний асистент. Відповідай коротко і по суті."),
        ("human", "{question}"),
    ]
)

chain = prompt | llm | StrOutputParser()


if __name__ == "__main__":
    questions = [
        "Привіт! Поясни простими словами, що таке LangChain.",
        "The future of artificial intelligence is",
        "Once upon a time in a distant galaxy",
        "The benefits of sustainable energy include",
    ]

    for question in questions:
        answer = chain.invoke({"question": question})
        print(answer)
