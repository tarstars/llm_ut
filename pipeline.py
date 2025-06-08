# pipeline.py

import json
import asyncio
import requests
from jinja2 import Template
import openai
from .models import PromptVersion, ResponseRecord

# LLM configuration
with open("oauth.txt", "r", encoding="utf-8") as f:
    _TOKEN = f.readline().strip()

_MODEL_URL = (
    "http://zeliboba.yandex-team.ru/balance/qwen3_23B_edu_ml/v1/chat/completions"
)

_HEADERS = {
    "Content-Type": "application/json",
    "X-Model-Discovery-Oauth-Token": _TOKEN,
    "Authorization": "Bearer EMPTY",
}


def call_validation_llm(messages, max_tokens: int = 32000, temperature: int = 0) -> str:
    """Send messages to the validation LLM and return the text response."""
    payload = {
        "model": "does_not_matter",
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(_MODEL_URL, headers=_HEADERS, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# load basket once
with open("basket.json") as f:
    BASKET = json.load(f)

# 1. generate_answers(prompt_text) → {case_id: answer_text}
# 2. evaluate_answers(answers) → {case_id: raw_eval_text}
# 3. parse_flags(raw_eval) → flags dict + final_flag
# 4. aggregate_metrics(all_flags) → metrics_summary
