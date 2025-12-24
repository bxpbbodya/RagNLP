from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, candidates):
        pairs = [(query, c[0]) for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [c for c, _ in ranked]
