import os
import re
import random

# ----------------------
# Налаштування
# ----------------------
DOCS_DIR = "data/docs"
CHUNK_DIR = "data/docs_chunks"
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)

CHUNK_SIZE = 120  # слів на один чанок
OVERLAP = 50  # перекриття слів

# Теми та підтемки для збільшення кількості документів
topics = {
    "Tokenization": ["Basics", "Subword Tokenization", "BPE", "WordPiece", "Char-level"],
    "Word Embeddings": ["Word2Vec", "GloVe", "FastText", "Contextual Embeddings", "Cosine Similarity"],
    "Language Modeling": ["Causal LM", "Masked LM", "Perplexity", "Applications", "Training Tips"],
    "Attention & Transformers": ["Self-Attention", "Multi-Head", "Positional Encoding", "Transformer Variants",
                                 "Scaling"],
    "Sequence Labeling (NER, POS)": ["NER Basics", "POS Tagging", "CRF Layer", "BiLSTM-CRF", "Transformer-based NER"],
    "Text Classification": ["Pipeline", "TF-IDF", "BERT Fine-Tuning", "Imbalanced Classes", "Evaluation"],
    "Evaluation Metrics": ["F1", "Accuracy", "BLEU", "ROUGE", "Precision-Recall Tradeoff"],
    "Overfitting & Regularization": ["Dropout", "L2 Regularization", "Early Stopping", "Data Augmentation",
                                     "Hyperparameter Tuning"],
    "Data Leakage & Train/Test Split": ["Data Leakage Examples", "Cross-validation", "Splits", "Preventing Leakage",
                                        "Real Cases"],
    "Prompting & RAG basics": ["RAG Concept", "Retrieval", "Generation", "Chunking", "Hallucinations"]
}


# ----------------------
# Функції
# ----------------------
def sanitize_filename(name: str) -> str:
    """Перетворює назву у безпечне для Windows ім'я файлу"""
    name = re.sub(r"[^\w\- ]", "_", name)
    name = name.replace(" ", "_")
    return name


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Розбиває текст на чанки"""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


# ----------------------
# Генерація тексту та документів
# ----------------------
sample_text_template = """
{} — це важлива тема NLP. У цьому розділі пояснюються ключові концепції та терміни.
Надаються приклади та порівняння підходів для кращого розуміння теми {}.
Текст подається українською мовою та містить достатньо інформації для chunking.
""".strip()

total_chunks = 0
doc_count = 0

for topic, subtopics in topics.items():
    for sub in subtopics:
        # Створюємо довгий текст для кожного документа
        repeat_paragraphs = random.randint(60, 80)  # ~50 слів на абзац
        text = "\n\n".join([sample_text_template.format(f"{topic} - {sub}", f"{topic} - {sub}")
                            for _ in range(repeat_paragraphs)])

        # Створюємо безпечне ім'я файлу
        fname = sanitize_filename(f"{topic}_{sub}") + ".md"
        path = os.path.join(DOCS_DIR, fname)

        # Записуємо сирий документ
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

        # Розбиваємо на чанки
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            out_path = os.path.join(CHUNK_DIR, f"{fname}_chunk{i}.txt")
            with open(out_path, "w", encoding="utf-8") as f_out:
                f_out.write(c)
        total_chunks += len(chunks)
        doc_count += 1

print(f"Створено {doc_count} документів у {DOCS_DIR}")
print(f"Створено {total_chunks} чанків у {CHUNK_DIR}")
