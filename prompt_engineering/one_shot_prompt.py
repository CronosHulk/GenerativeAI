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
            ("Here is an example of translating a sentence from English to French:"),

            (" English: How is the weather today?'"),
            (" French: “Comment est le temps aujourd'hui?”"),

            (" Now, translate the following sentence from English to French:"),
            (" English: “{question}”"),
        ]
)

chain = prompt | llm | StrOutputParser()


if __name__ == "__main__":
    question = "Hi! What is LangChain?"

    answer = chain.invoke({"question": question})
    print(answer)
