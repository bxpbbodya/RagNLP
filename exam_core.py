import json
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

DATA_DIR = "data"
EXAMS_PATH = os.path.join(DATA_DIR, "exams.jsonl")
STUDENTS_PATH = "students.json"

ALL_TOPICS = [
    "Tokenization",
    "Word Embeddings",
    "Language Modeling",
    "Attention & Transformers",
    "Sequence Labeling (NER, POS)",
    "Text Classification",
    "Evaluation Metrics (BLEU, ROUGE, F1)",
    "Overfitting & Regularization",
    "Data Leakage & Train/Test Split",
    "Prompting & RAG basics"
]

TOPIC_QUESTION_BANK = {
    "Tokenization": [
        "Що таке токенізація? Порівняй word-level, subword (BPE/WordPiece) та char-level.",
        "Чому subword токенізація часто краща за word-level у реальних NLP задачах?"
    ],
    "Word Embeddings": [
        "Що таке word embeddings? Чим Word2Vec/GloVe відрізняються від contextual embeddings?",
        "Поясни, що таке cosine similarity і навіщо вона в embeddings."
    ],
    "Language Modeling": [
        "Що таке мовна модель? Як відрізняються causal LM і masked LM?",
        "Що таке perplexity і як її інтерпретувати?"
    ],
    "Attention & Transformers": [
        "Поясни self-attention: запити/ключі/значення (Q/K/V) на інтуїтивному рівні.",
        "Чому трансформери краще масштабується, ніж RNN, для довгих контекстів?"
    ],
    "Sequence Labeling (NER, POS)": [
        "Що таке NER/POS як sequence labeling? Як це відрізняється від classification?",
        "Навіщо інколи додають CRF поверх BiLSTM/Transformer?"
    ],
    "Text Classification": [
        "Як будується pipeline для текстової класифікації? (дані → фічі/модель → оцінка)",
        "Чому важливо балансувати класи і що робити при дисбалансі?"
    ],
    "Evaluation Metrics (BLEU, ROUGE, F1)": [
        "Коли доречний F1, а коли accuracy? Поясни на прикладі дисбалансу класів.",
        "Чим BLEU відрізняється від ROUGE і де що застосовують?"
    ],
    "Overfitting & Regularization": [
        "Що таке overfitting? Як його помітити по train/val метриках?",
        "Назви 3 способи регуляризації і коротко поясни кожен."
    ],
    "Data Leakage & Train/Test Split": [
        "Що таке data leakage? Наведи приклад і як уникати.",
        "Навіщо потрібен train/val/test split і що таке крос-валідація?"
    ],
    "Prompting & RAG basics": [
        "Поясни базовий принцип RAG: retrieval + generation. Навіщо це треба?",
        "Які типові провали RAG (не той контекст, hallucinations) і як їх зменшувати?"
    ]
}

RUBRIC = """
Оціни відповідь студента по 0..10.
10: точна, повна, з прикладами/термінами, без грубих помилок.
7-9: в цілому правильно, але є прогалини/мало прикладів.
4-6: частково правильно, плутанина, пропущені ключові моменти.
1-3: майже не по темі або багато помилок.
0: "не знаю" або порожньо.
Поверни ТІЛЬКИ число 0..10 та один рядок короткого фідбеку (1-2 речення).
Формат:
score: X
feedback: ...
""".strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_students() -> List[Dict]:
    if not os.path.exists(STUDENTS_PATH):
        return []
    with open(STUDENTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _append_jsonl(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


@dataclass
class Message:
    role: str  # "system" | "user" | "tool_call"
    content: str
    datetime: str


@dataclass
class ExamState:
    email: str
    name: str
    topics: List[str]
    current_topic_idx: int = 0
    topic_turns: int = 0
    total_score: float = 0.0
    asked_questions: int = 0
    finished: bool = False


# -------------------------
# TOOLS (за умовою завдання)
# -------------------------

def start_exam(email: str, name: str) -> List[str]:
    """
    Викликається, коли бот має достатньо інформації про студента.
    Перевіряє, що студент існує у "базі" students.json.
    Пише запис про старт і повертає 2-3 випадкові теми.
    """
    students = _load_students()
    exists = any(s["email"].strip().lower() == email.strip().lower() and
                 s["name"].strip().lower() == name.strip().lower()
                 for s in students)
    if not exists:
        raise ValueError("Student not found in database. Please re-check name/email.")

    topics = random.sample(ALL_TOPICS, k=random.choice([2, 3]))
    _append_jsonl(EXAMS_PATH, {
        "event": "start_exam",
        "email": email,
        "name": name,
        "topics": topics,
        "datetime": _now_iso()
    })
    return topics


def get_next_topic(state: ExamState) -> str:
    """
    Викликається, коли потрібна нова тема для обговорення.
    """
    if state.current_topic_idx >= len(state.topics):
        return ""
    topic = state.topics[state.current_topic_idx]
    return topic


def end_exam(email: str, score: float, history: List[Dict]) -> None:
    """
    Записує результат іспиту. history — повна історія чату.
    """
    _append_jsonl(EXAMS_PATH, {
        "event": "end_exam",
        "email": email,
        "score": score,
        "history": history,
        "datetime": _now_iso()
    })


# -------------------------
# LLM helper (optional)
# -------------------------

def call_openai_compatible_chat(
    api_key: str,
    base_url: str,
    model: str,
    messages: List[Dict],
    timeout: int = 40
) -> str:
    """
    Працює з будь-яким OpenAI-compatible Chat Completions endpoint:
    POST {base_url}/chat/completions
    """
    import requests

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def grade_answer(
    topic: str,
    question: str,
    answer: str,
    use_llm: bool,
    api_key: str,
    base_url: str,
    model: str
) -> Tuple[float, str]:
    """
    Повертає (score, feedback).
    Якщо LLM недоступний — евристичний скоринг.
    """
    if not answer.strip() or "не знаю" in answer.strip().lower():
        return 0.0, "Ок, прийнято. Якщо хочеш — скажи, що саме не зрозуміло, і я підкажу після іспиту."

    if use_llm and api_key.strip():
        prompt = [
            {"role": "system", "content": "Ти суворий, але чесний екзаменатор з NLP."},
            {"role": "user", "content": f"Тема: {topic}\nПитання: {question}\nВідповідь студента: {answer}\n\n{RUBRIC}"}
        ]
        raw = call_openai_compatible_chat(api_key, base_url, model, prompt)
        # простий парсер формату
        score = 0.0
        feedback = raw.strip()
        for line in raw.splitlines():
            if line.lower().startswith("score:"):
                try:
                    score = float(line.split(":", 1)[1].strip())
                except:
                    pass
            if line.lower().startswith("feedback:"):
                feedback = line.split(":", 1)[1].strip()
        score = max(0.0, min(10.0, score))
        return score, feedback if feedback else "Дякую, рухаємось далі."
    else:
        # OFFLINE евристика: ключові слова + структура
        keywords = {
            "Tokenization": ["bpe", "wordpiece", "subword", "токен", "vocab"],
            "Word Embeddings": ["embedding", "word2vec", "glove", "cosine", "context"],
            "Language Modeling": ["perplexity", "masked", "causal", "lm", "ймовір"],
            "Attention & Transformers": ["attention", "q", "k", "v", "self-attention", "transformer"],
            "Sequence Labeling (NER, POS)": ["ner", "pos", "sequence", "crf", "tag"],
            "Text Classification": ["classification", "labels", "tf-idf", "bag", "fine-tune"],
            "Evaluation Metrics (BLEU, ROUGE, F1)": ["f1", "precision", "recall", "bleu", "rouge", "accuracy"],
            "Overfitting & Regularization": ["overfit", "dropout", "l2", "regular", "early stopping"],
            "Data Leakage & Train/Test Split": ["leakage", "split", "cross-validation", "train", "test"],
            "Prompting & RAG basics": ["rag", "retrieval", "context", "chunk", "vector"]
        }
        hits = 0
        low = answer.lower()
        for kw in keywords.get(topic, []):
            if kw in low:
                hits += 1
        length_bonus = 1 if len(answer.strip()) > 250 else 0
        score = min(10.0, 3.0 + hits * 1.3 + length_bonus * 1.2)
        feedback = "В цілому ок. Додай більше чітких визначень і 1-2 приклади/порівняння."
        if score >= 8:
            feedback = "Добре! Відповідь виглядає впевнено. Можна ще додати приклад для закріплення."
        elif score <= 5:
            feedback = "Є частково правильні думки, але бракує ключових термінів/структури. Спробуй чіткіше."
        return float(round(score, 1)), feedback


def pick_question(topic: str, asked_questions: List[str]) -> str:
    bank = TOPIC_QUESTION_BANK.get(topic, [])
    if not bank:
        return f"Поясни тему: {topic} своїми словами та наведи приклад."
    # не повторюватися
    candidates = [q for q in bank if q not in asked_questions]
    if candidates:
        return random.choice(candidates)
    return random.choice(bank)


def make_final_feedback(avg_score: float) -> str:
    if avg_score >= 8.5:
        return "Сильний результат: відповіді здебільшого точні, з правильними термінами. Підтягни дрібні деталі та приклади."
    if avg_score >= 6.5:
        return "Нормальний рівень: базові поняття є, але місцями не вистачає глибини/прикладів. Рекомендую повторити слабкі теми."
    if avg_score >= 4.5:
        return "Початковий рівень: є правильні фрагменти, але багато прогалин. Варто системно пройти конспект + зробити практику."
    return "Поки слабко: треба відновити базу з ключових тем. Почни з токенізації, embeddings і метрик, і рухайся далі."
