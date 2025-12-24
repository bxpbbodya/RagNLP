---
title: RagNLP
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# RagNLP — RAG Question Answering українською

Це демо Retrieval-Augmented Generation (RAG) системи для question answering на основі документів.

## Як користуватися
1. Введіть запитання.
2. Оберіть режим retriever (BM25 / Dense / Combined).
3. (Опційно) увімкніть реранкер.
4. Додайте API key (Groq або інший OpenAI-compatible).
5. Натисніть Ask.

## Компоненти
- **Джерело даних:** українські документи по NLP/RAG
- **Chunking:** фіксовані чанки з overlap
- **Retrievers:** BM25 + Dense
- **Reranker:** Cross-encoder
- **UI:** Gradio
- **Citations:** inline + список джерел в кінці
