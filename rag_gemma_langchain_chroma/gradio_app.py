import gradio as gr

from prepare_documents import ingest
from rag_pipeline import ask, format_sources
from settings import DEFAULT_SOURCE_URL, env


DEFAULT_URL = env("RAG_SOURCE_URL", DEFAULT_SOURCE_URL)


def show_question_view(url: str) -> tuple:
    if not url.strip():
        return (
            "URL is required.",
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    try:
        chunk_count = ingest(url=url.strip())
    except Exception as error:
        return (
            f"Failed to prepare document: {error}",
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    return (
        f"Document indexed. Chunks: {chunk_count}",
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
    )


def show_answer_view(question: str) -> tuple:
    if not question.strip():
        return (
            "Question is required.",
            "",
            gr.update(visible=True),
            gr.update(visible=False),
        )

    try:
        result = ask(question.strip())
    except Exception as error:
        return (
            f"Failed to answer: {error}",
            "",
            gr.update(visible=True),
            gr.update(visible=False),
        )

    sources = format_sources(result["source_documents"])
    return (
        result["answer"],
        sources,
        gr.update(visible=False),
        gr.update(visible=True),
    )


def ask_another_question() -> tuple:
    return (
        "",
        "",
        "",
        gr.update(visible=True),
        gr.update(visible=False),
    )


with gr.Blocks(title="Gemma RAG") as demo:
    gr.Markdown("# Gemma RAG")

    with gr.Group(visible=True) as load_view:
        source_url = gr.Textbox(
            label="Document URL",
            value=DEFAULT_URL,
            lines=2,
        )
        load_button = gr.Button("Завантажити файл за посиланням", variant="primary")
        load_status = gr.Markdown()

    with gr.Group(visible=False) as question_view:
        question = gr.Textbox(
            label="Питання",
            placeholder="Ask something about the indexed document",
            lines=3,
        )
        ask_button = gr.Button("Запитати", variant="primary")

    with gr.Group(visible=False) as answer_view:
        answer = gr.Markdown(label="Відповідь")
        sources = gr.Textbox(label="Sources", lines=12)
        another_button = gr.Button("Спитати щось ще")

    load_button.click(
        fn=show_question_view,
        inputs=source_url,
        outputs=[load_status, load_view, question_view, answer_view],
    )
    ask_button.click(
        fn=show_answer_view,
        inputs=question,
        outputs=[answer, sources, question_view, answer_view],
    )
    another_button.click(
        fn=ask_another_question,
        outputs=[question, answer, sources, question_view, answer_view],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
