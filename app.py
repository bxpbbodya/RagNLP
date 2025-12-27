import gradio as gr
from rag_pipeline import RAGPipeline

rag = RAGPipeline()

# --- Константи для провайдерів ---
PROVIDERS = {
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "note": "Використовується Groq API key."
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "note": "Використовується OpenAI API key."
    },
    "Custom": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "note": "OpenAI-compatible endpoint. Base URL вводиться вручну."
    }
}

DOMAIN_TEXT = """
# 📚 RAG Question Answering (NLP)

Це демо RAG (Retrieval-Augmented Generation) системи Question Answering на базі українських документів **з тем NLP та RAG**.

**Покриті теми:**
- Tokenization (BPE / WordPiece / char-level)
- Word Embeddings (Word2Vec, GloVe, cosine similarity)
- Transformers (self-attention, Q/K/V, positional encoding)
- Evaluation Metrics (F1, BLEU, ROUGE)
- Overfitting, Data leakage, Train/Val/Test split
- Prompting та RAG basics

**Приклади запитів:**
- “Поясни self-attention: що таке Q, K, V?”
- “Чим BLEU відрізняється від ROUGE?”
- “Що таке data leakage і як уникати?”
- “Що таке cosine similarity і навіщо вона для embeddings?”
- “Які типові провали RAG і як їх зменшувати?”

Якщо питання **не покривається документами**, система повідомить: **“Немає інформації в документах.”**
""".strip()


def on_provider_change(provider: str):
    cfg = PROVIDERS.get(provider, PROVIDERS["Groq"])
    # base_url visible тільки для Custom
    base_visible = (provider == "Custom")
    return (
        gr.update(value=cfg["base_url"], visible=base_visible),
        gr.update(value=cfg["model"]),
        gr.update(value=cfg["note"])
    )


def ask(question, use_bm25, use_dense, api_key, provider, base_url, model):
    try:
        # якщо не Custom — беремо base_url з provider конфігів
        if provider in PROVIDERS and provider != "Custom":
            base_url = PROVIDERS[provider]["base_url"]

        answer, sources = rag.answer(
            question=question,
            use_bm25=use_bm25,
            use_dense=use_dense,
            api_key=api_key,
            base_url=base_url,
            model=model
        )

        src_text = "\n".join([f"[{i+1}] {s}" for i, s in enumerate(sources)])
        return answer, src_text
    except Exception as e:
        return f"❌ Помилка: {str(e)}", ""


with gr.Blocks(title="RAG NLP QA") as demo:
    gr.Markdown(DOMAIN_TEXT)

    question = gr.Textbox(
        label="Question",
        placeholder="Наприклад: Поясни self-attention: що таке Q, K, V?"
    )

    with gr.Row():
        use_bm25 = gr.Checkbox(label="BM25", value=True)
        use_dense = gr.Checkbox(label="Semantic", value=True)

    gr.Markdown("### LLM settings")
    api_key = gr.Textbox(label="API key", type="password")

    provider = gr.Dropdown(
        label="Provider",
        choices=["Groq", "OpenAI", "Custom"],
        value="Groq"
    )

    base_url = gr.Textbox(
        label="Base URL (тільки для Custom)",
        value=PROVIDERS["Groq"]["base_url"],
        visible=False
    )

    model = gr.Textbox(label="Model", value=PROVIDERS["Groq"]["model"])
    provider_note = gr.Markdown(PROVIDERS["Groq"]["note"])

    answer = gr.Textbox(label="Answer", lines=6)
    sources = gr.Textbox(label="Sources", lines=6)

    btn = gr.Button("Ask")

    provider.change(
        on_provider_change,
        inputs=[provider],
        outputs=[base_url, model, provider_note]
    )

    btn.click(
        ask,
        inputs=[question, use_bm25, use_dense, api_key, provider, base_url, model],
        outputs=[answer, sources]
    )

if __name__ == "__main__":
    demo.launch()
