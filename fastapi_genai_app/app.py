from fastapi import FastAPI
from pydantic import BaseModel, Field

from model import generate_json, generate_text


app = FastAPI(title="Gemma GenAI API", version="1.0.0")


class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


class TextResponse(BaseModel):
    response: str


@app.get("/health")
def health():
    return {"status": "ok", "model": "gemma"}


@app.post("/generate", response_model=TextResponse)
def generate(request: PromptRequest):
    response = generate_text(request.prompt)
    return TextResponse(response=response)


@app.post("/generate-json")
def generate_structured_json(request: PromptRequest):
    return generate_json(request.prompt)
