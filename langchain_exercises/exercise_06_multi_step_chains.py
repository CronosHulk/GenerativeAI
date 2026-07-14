from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel

from common import make_ollama_chat, print_section


llm = make_ollama_chat(temperature=0.2)

sentiment_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Classify product review sentiment. Return one word only."),
        ("human", "Review:\n{review}\n\nSentiment:"),
    ]
)

summary_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Summarize product feedback in one short sentence."),
        ("human", "Review:\n{review}"),
    ]
)

response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write a helpful customer support response based on the analysis.",
        ),
        (
            "human",
            """Review:
{review}

Sentiment:
{sentiment}

Summary:
{summary}

Response:""",
        ),
    ]
)

sentiment_chain = sentiment_prompt | llm | StrOutputParser()
summary_chain = summary_prompt | llm | StrOutputParser()
response_chain = response_prompt | llm | StrOutputParser()


def sequential_review_chain(review: str) -> dict[str, str]:
    sentiment = sentiment_chain.invoke({"review": review})
    summary = summary_chain.invoke({"review": review})
    response = response_chain.invoke(
        {
            "review": review,
            "sentiment": sentiment,
            "summary": summary,
        }
    )
    return {
        "sentiment": sentiment,
        "summary": summary,
        "response": response,
    }


parallel_analysis = RunnableParallel(
    {
        "sentiment": sentiment_chain,
        "summary": summary_chain,
    }
)


def add_review(inputs: dict) -> dict:
    return {
        "review": inputs["review"],
        "sentiment": inputs["analysis"]["sentiment"],
        "summary": inputs["analysis"]["summary"],
    }


lcel_review_chain = (
    RunnableParallel(
        {
            "review": RunnableLambda(lambda x: x["review"]),
            "analysis": parallel_analysis,
        }
    )
    | RunnableLambda(add_review)
    | RunnableParallel(
        {
            "sentiment": RunnableLambda(lambda x: x["sentiment"]),
            "summary": RunnableLambda(lambda x: x["summary"]),
            "response": response_chain,
        }
    )
)


if __name__ == "__main__":
    sample_review = (
        "The laptop is fast and the screen is beautiful, but the fan becomes loud "
        "during video calls and the battery drains much faster than advertised."
    )

    print_section("Traditional sequential approach")
    print(sequential_review_chain(sample_review))

    print_section("Modern LCEL approach with parallel analysis")
    print(lcel_review_chain.invoke({"review": sample_review}))

    print_section("Comparison notes")
    print(
        """
Sequential function:
- Easy to read step by step.
- Good when every step depends on previous output.

LCEL pipeline:
- More composable.
- RunnableParallel can run independent sentiment and summary steps together.
- Easier to reuse inside larger LangChain applications.
"""
    )
