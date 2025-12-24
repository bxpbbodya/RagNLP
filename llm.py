import requests


def call_llm(api_key, base_url, model, prompt):
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Ти корисний асистент з NLP."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
