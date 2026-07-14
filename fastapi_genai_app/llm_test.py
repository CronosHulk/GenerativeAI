from model import generate_json, generate_text


if __name__ == "__main__":
    prompt = "What is the capital of Canada? Add one interesting fact."

    print("Text response:")
    print(generate_text(prompt))

    print("\nStructured JSON response:")
    print(generate_json(prompt))
