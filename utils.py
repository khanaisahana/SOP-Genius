# Send to OpenRouter

import requests

def query_llm_openrouter(prompt, api_key, referer_url):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Referer": referer_url,
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]
