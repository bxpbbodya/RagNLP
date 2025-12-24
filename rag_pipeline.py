import os
from chunking import chunk_text
from retrievers import BM25Retriever, DenseRetriever
from reranker import Reranker
from llm import call_llm


def load_documents(path="data/docs"):
    docs = []
    meta = []
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
        query,
        use_bm25=True,
        use_dense=True,
        api_key="",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini"
    ):
        candidates = []

        if use_bm25:
            candidates += self.bm25.search(query)

        if use_dense:
            candidates += self.dense.search(query)

        if not candidates:
            return "Пошук вимкнено — немає контексту.", []

        reranked = self.reranker.rerank(query, candidates)[:5]

        context = "\n\n".join([c[0] for c in reranked])

        prompt = f"""
Використай ТІЛЬКИ інформацію з контексту нижче.
Якщо відповіді немає — скажи "Немає інформації в документах".

Контекст:
{context}

Питання:
{query}
"""

        answer = call_llm(api_key, base_url, model, prompt)

        sources = [self.meta[c[2]] for c in reranked]

        return answer, sources
