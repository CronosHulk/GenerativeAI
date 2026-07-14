from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore

from common import HashEmbeddings, format_docs, make_ollama_chat, print_section
from exercise_03_document_loaders_splitters import split_documents


DATA_DIR = Path(__file__).parent / "data"


def load_ai_document() -> list[Document]:
    path = DATA_DIR / "ai_overview.txt"
    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": str(path)},
        )
    ]


def build_retriever():
    docs = load_ai_document()
    chunks = split_documents(docs, chunk_size=450, chunk_overlap=75)
    vector_store = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=HashEmbeddings(),
    )
    return vector_store.as_retriever(search_kwargs={"k": 3})


def build_qa_chain():
    retriever = build_retriever()
    llm = make_ollama_chat(temperature=0.1)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer using only the provided context. If the context is not enough, say so.",
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}""",
            ),
        ]
    )

    retrieval_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    return retrieval_chain | prompt | llm | StrOutputParser()


if __name__ == "__main__":
    qa_chain = build_qa_chain()

    questions = [
        "What is retrieval augmented generation?",
        "What are common neural network components?",
        "Which responsible AI practices are mentioned?",
    ]

    for question in questions:
        print_section(question)
        print(qa_chain.invoke(question))
