from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util


class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        tokenized = [c.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, k=5):
        scores = self.bm25.get_scores(query.lower().split())
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i]), i) for i in top]


class DenseRetriever:
    def __init__(self, chunks):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks = chunks
        self.embeddings = self.model.encode(chunks, convert_to_tensor=True)

    def search(self, query, k=5):
        q_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, self.embeddings)[0]
        top = scores.topk(k)
        return [
            (self.chunks[i], float(scores[i]), int(i))
            for i in top.indices
        ]
