import json
import asyncio
import requests
from jinja2 import Template
from typing import Dict, List
from sqlmodel import select
from app.model import PromptVersion, ResponseRecord
from app.database import get_session, AsyncSession

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

with open("basket.json") as f:
    BASKET = json.load(f)

def simple_evaluate(answer: str) -> str:
    correctness = "Yes" if "return" in answer else "No"
    code_presence = "Yes" if "def" in answer else "No"
    line_reference = "Yes" if "line" in answer else "No"
    return (
        f"<correctness>{correctness}</correctness>"
        f"<code_presence>{code_presence}</code_presence>"
        f"<line_reference>{line_reference}</line_reference>"
    )

def parse_flags(raw: str) -> Dict[str, bool]:
    def extract(tag: str) -> bool:
        open_t = f"<{tag}>"
        close_t = f"</{tag}>"
        if open_t in raw and close_t in raw:
            val = raw.split(open_t)[1].split(close_t)[0].strip()
            return val.lower() == "yes"
        return False
    return {
        "correctness": extract("correctness"),
        "code_presence": extract("code_presence"),
        "line_reference": extract("line_reference"),
    }

async def process_prompt(prompt_id: str, text: str):
    template = Template(text)
    async with get_session() as session:
        responses: List[ResponseRecord] = []
        for case in BASKET:
            answer = template.render(**case)
            raw_eval = simple_evaluate(answer)
            flags = parse_flags(raw_eval)
            final_flag = all(flags.values())
            rec = ResponseRecord(
                prompt_id=prompt_id,
                case_id=case["id"],
                answer=answer,
                raw_evaluation=raw_eval,
                flags=flags,
                final_flag=final_flag,
            )
            session.add(rec)
            responses.append(rec)
        await session.commit()
        if responses:
            metrics = {
                "correctness": sum(r.flags["correctness"] for r in responses) / len(responses),
                "code_presence": sum(r.flags["code_presence"] for r in responses) / len(responses),
                "line_reference": sum(r.flags["line_reference"] for r in responses) / len(responses),
            }
            metrics["final_score"] = sum(r.final_flag for r in responses) / len(responses)
            result = await session.exec(select(PromptVersion).where(PromptVersion.prompt_id == prompt_id))
            pv = result.one()
            pv.metrics = metrics
            session.add(pv)
            await session.commit()
