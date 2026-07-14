from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate


json_parser = JsonOutputParser()

text_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer clearly, accurately, and concisely.",
        ),
        ("human", "{user_prompt}"),
    ]
)

json_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that returns only valid JSON.",
        ),
        (
            "human",
            """Analyze the user request and return a structured response.

User request:
{user_prompt}

Return a JSON object with these fields:
- answer: string
- category: string
- confidence: number between 0 and 1
- next_steps: array of strings

{format_instructions}""",
        ),
    ]
).partial(format_instructions=json_parser.get_format_instructions())
