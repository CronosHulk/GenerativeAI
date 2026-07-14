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
            ("Here are few examples of classifying emotions in statements:"),

            ("Statement: 'I just won my first marathon!'"),
            ("Emotion: Joy"),

            ("Statement: 'I can't believe I lost my keys again.'"),
            ("Emotion: Frustration"),

            ("Statement: 'My best friend is moving to another country.'"),
            ("Emotion: Sadness"),

            ("Now, classify the emotion in the following statement:"),
            ("Statement: '{question}'")
        ]
)

chain = prompt | llm | StrOutputParser()


if __name__ == "__main__":
    question = "That movie was so scary I had to cover my eyes"

    answer = chain.invoke({"question": question})
    print(answer)
