from flask import Flask, jsonify, request

from model import generate_json, generate_text


app = Flask(__name__)


def get_user_prompt() -> tuple[str | None, tuple | None]:
    data = request.get_json(silent=True) or {}
    user_prompt = data.get("prompt") or data.get("user_prompt")

    if not user_prompt:
        return None, (jsonify({"error": "Field 'prompt' is required."}), 400)

    return user_prompt, None


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": "gemma"})


@app.post("/generate")
def generate():
    user_prompt, error = get_user_prompt()
    if error:
        return error

    response = generate_text(user_prompt)
    return jsonify({"response": response})


@app.post("/generate-json")
def generate_structured_json():
    user_prompt, error = get_user_prompt()
    if error:
        return error

    response = generate_json(user_prompt)
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
