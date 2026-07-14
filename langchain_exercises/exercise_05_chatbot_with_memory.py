from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from common import make_ollama_chat, print_section


store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


llm = make_ollama_chat(temperature=0.3)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a concise assistant. Use the conversation history when it helps.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

chain = prompt | llm | StrOutputParser()

chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


def ask(message: str, session_id: str = "demo") -> str:
    return chatbot.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}},
    )


if __name__ == "__main__":
    messages = [
        "My name is Alex and I am learning LangChain.",
        "I prefer short technical explanations.",
        "What am I learning?",
        "How should you explain it to me?",
    ]

    for message in messages:
        print_section(f"User: {message}")
        print(ask(message))

    print_section("Stored conversation history")
    for item in get_session_history("demo").messages:
        print(f"{item.type}: {item.content}")
