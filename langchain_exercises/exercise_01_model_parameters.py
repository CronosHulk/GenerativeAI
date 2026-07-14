from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from common import DEFAULT_OLLAMA_MODEL, make_ollama_chat, print_section


PROMPTS = {
    "Creative writing": "Write a short poem about artificial intelligence.",
    "Factual question": "What are the key components of a neural network?",
    "Instruction-following": "List 5 tips for effective time management.",
}


def build_chain(temperature: float):
    llm = make_ollama_chat(temperature=temperature)
    prompt = PromptTemplate.from_template("{prompt}")
    return prompt | llm | StrOutputParser()


def run_comparison() -> None:
    model_configs = [
        (f"{DEFAULT_OLLAMA_MODEL} low temperature", 0.2),
        (f"{DEFAULT_OLLAMA_MODEL} high temperature", 0.8),
    ]

    chains = [
        (label, temperature, build_chain(temperature))
        for label, temperature in model_configs
    ]

    for prompt_type, prompt_text in PROMPTS.items():
        print_section(prompt_type)
        print(f"Prompt: {prompt_text}")

        for label, temperature, chain in chains:
            print_section(f"{label} | temperature={temperature}")
            for run_number in range(1, 3):
                response = chain.invoke({"prompt": prompt_text})
                print(f"\nRun {run_number}:\n{response}")


def print_observation_template() -> None:
    print_section("Observation checklist")
    print(
        """
Document your observations:

1. Creativity compared to consistency:
   - Low temperature should usually produce more direct and stable answers.
   - High temperature should usually produce more varied wording and ideas.

2. Variation between multiple runs:
   - Compare Run 1 and Run 2 for each model and prompt type.
   - Creative writing should show the most visible variation.

3. Appropriateness for different tasks:
   - Use lower temperature for factual and instruction-following tasks.
   - Use higher temperature when exploration, style, or brainstorming matters.
"""
    )


if __name__ == "__main__":
    run_comparison()
    print_observation_template()
