from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common import make_ollama_chat, print_section


parser = JsonOutputParser()
llm = make_ollama_chat(temperature=0.1)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured information and return only valid JSON.",
        ),
        (
            "human",
            """Analyze the following product feedback.

Feedback:
{feedback}

Return a JSON object with these fields:
- product: string
- sentiment: one of positive, neutral, negative
- main_issue: string or null
- requested_action: string
- priority: one of low, medium, high

{format_instructions}""",
        ),
    ]
).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser


def run_examples() -> None:
    feedback_items = [
        "The wireless headphones sound great, but the battery only lasts two hours.",
        "The project management app is clean, fast, and easy for my team to adopt.",
        "The payment terminal crashes every morning during checkout. Please fix it urgently.",
    ]

    for feedback in feedback_items:
        print_section("Input")
        print(feedback)

        result = chain.invoke({"feedback": feedback})
        print_section("Parsed dictionary")
        print(result)

        print_section("Specific fields")
        print(f"Product: {result['product']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Priority: {result['priority']}")
        print(f"Is Python dict: {isinstance(result, dict)}")


if __name__ == "__main__":
    run_examples()
