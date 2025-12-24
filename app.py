import gradio as gr
from rag_pipeline import RAGPipeline

rag = RAGPipeline()


def ask(
    question,
    use_bm25,
    use_dense,
    api_key,
    base_url,
    model
):
    answer, sources = rag.answer(
        question,
        use_bm25,
        use_dense,
        api_key,
        base_url,
        model
    )

    src_text = "\n".join([f"[{i+1}] {s}" for i, s in enumerate(sources)])
    return answer, src_text


with gr.Blocks(title="RAG NLP QA") as demo:
    gr.Markdown("# 📚 RAG Question Answering (NLP)")

    question = gr.Textbox(label="Question")

    with gr.Row():
        use_bm25 = gr.Checkbox(label="BM25", value=True)
        use_dense = gr.Checkbox(label="Semantic", value=True)

    gr.Markdown("### LLM settings")
    api_key = gr.Textbox(label="API key", type="password")
    base_url = gr.Textbox(label="Base URL", value="https://api.openai.com/v1")
    model = gr.Textbox(label="Model", value="gpt-4o-mini")

    answer = gr.Textbox(label="Answer", lines=5)
    sources = gr.Textbox(label="Sources")

    btn = gr.Button("Ask")

    btn.click(
        ask,
        inputs=[question, use_bm25, use_dense, api_key, base_url, model],
        outputs=[answer, sources]
    )

if __name__ == "__main__":
    demo.launch()
