import os
from chunking import chunk_text
from retrievers import BM25Retriever, DenseRetriever
from reranker import Reranker
from llm import call_llm


def load_documents(path="data/docs"):
    docs = []
    meta = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Docs folder not found: {path}")

    for fname in os.listdir(path):
        if fname.endswith(".md"):
            with open(os.path.join(path, fname), encoding="utf-8") as f:
                text = f.read()
                chunks = chunk_text(text)
                for i, c in enumerate(chunks):
                    docs.append(c)
                    meta.append(f"{fname} — chunk {i}")
    return docs, meta


class RAGPipeline:
    def __init__(self):
        self.chunks, self.meta = load_documents()
        self.bm25 = BM25Retriever(self.chunks)
        self.dense = DenseRetriever(self.chunks)
        self.reranker = Reranker()

    def answer(
        self,
        question: str,
        use_bm25: bool = True,
        use_dense: bool = True,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini"
    ):
        query = (question or "").strip()
        if not query:
            return "❌ Введіть питання.", []

        candidates = []

        if use_bm25:
            candidates += self.bm25.search(query)

        if use_dense:
            candidates += self.dense.search(query)

        if not candidates:
            return "Пошук вимкнено або не знайдено релевантного контексту.", []

        # Реранк і топ-5
        reranked = self.reranker.rerank(query, candidates)[:5]

        # Контекст і джерела
        context_blocks = []
        sources = []
        for c in reranked:
            chunk_text_str = c[0]
            idx = c[2]
            context_blocks.append(chunk_text_str)
            sources.append(self.meta[idx])

        context = "\n\n".join([f"[{i+1}] {t}" for i, t in enumerate(context_blocks)])

        # Якщо немає ключа — робимо retrieval-only mode (не падає)
        if not api_key.strip():
            preview = "\n\n".join([f"[{i+1}] {t[:350]}..." for i, t in enumerate(context_blocks)])
            return (
                "⚠️ API key не введено, тому генерація відповіді вимкнена.\n"
                "✅ Але retrieval працює — ось топ релевантні фрагменти:\n\n"
                f"{preview}",
                sources
            )

        # --- М’якший промпт (виправляє проблему “нема інформації”, коли вона є) ---
        prompt = f"""
Ти — асистент для Question Answering на базі RAG.

ПРАВИЛА:
1) Відповідай ТІЛЬКИ на основі контексту.
2) Якщо у контексті є релевантна інформація — сформуй відповідь, МОЖНА узагальнювати з кількох фрагментів.
3) Якщо контекст не містить відповіді — скажи "Немає інформації в документах." і коротко поясни, чому (наприклад, "у джерелах говориться про X, але не про Y").
4) Не вигадуй фактів поза контекстом.
5) За можливості додай короткі inline-цитати [1], [2] до ключових тверджень.

КОНТЕКСТ:
{context}

ПИТАННЯ:
{query}

ВІДПОВІДЬ:
""".strip()

        answer = call_llm(api_key, base_url, model, prompt)

        # safety: якщо call_llm повернув None/порожнє
        if not answer or not str(answer).strip():
            answer = "❌ Не вдалося отримати відповідь від LLM (порожня відповідь)."

        return answer, sources
