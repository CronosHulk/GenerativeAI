import ast
import operator

from langchain_core.tools import tool
from langchain.agents import create_agent

from common import make_ollama_chat, print_section


ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def safe_eval_math(expression: str) -> float:
    def eval_node(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[type(node.op)](
                eval_node(node.left),
                eval_node(node.right),
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATORS:
            return ALLOWED_OPERATORS[type(node.op)](eval_node(node.operand))
        raise ValueError("Only basic arithmetic expressions are allowed.")

    tree = ast.parse(expression, mode="eval")
    return eval_node(tree.body)


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression such as '12 * (4 + 3)'."""
    return str(safe_eval_math(expression))


@tool
def text_formatter(text: str, style: str = "title") -> str:
    """Format text. Styles: upper, lower, title, sentence."""
    if style == "upper":
        return text.upper()
    if style == "lower":
        return text.lower()
    if style == "sentence":
        return text[:1].upper() + text[1:].lower()
    return text.title()


def build_agent():
    llm = make_ollama_chat(temperature=0)
    return create_agent(
        model=llm,
        tools=[calculator, text_formatter],
        system_prompt="You are a helpful assistant. Use tools when calculations or text formatting are needed.",
    )


if __name__ == "__main__":
    agent = build_agent()

    questions = [
        "What is 18 * (7 + 5)? Use the calculator tool.",
        "Format 'langchain agents use tools' as title case. Use the text_formatter tool.",
        "Calculate 144 / 12, then format the sentence 'the answer is 12' in uppercase.",
    ]

    for question in questions:
        print_section(question)
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        print(result["messages"][-1].content)
