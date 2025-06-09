import json
import asyncio
try:
    import requests
except ModuleNotFoundError:  # allow running tests without dependency
    class requests:
        @staticmethod
        def post(*args, **kwargs):
            raise RuntimeError("requests library not installed")
try:
    from jinja2 import Template
except ModuleNotFoundError:  # allow running tests without dependency
    class Template:
        def __init__(self, text: str):
            self.text = text

        def render(self, **context) -> str:
            return self.text.format(**context)
from typing import Dict, List
try:
    from sqlmodel import select
    from app.model import PromptVersion, ResponseRecord
    from app.database import get_session, AsyncSession
except ModuleNotFoundError:  # allow running tests without dependency
    select = None
    PromptVersion = ResponseRecord = object
    AsyncSession = None

    async def get_session():
        raise RuntimeError("greenlet_spawn: sqlmodel not installed")
        yield  # pragma: no cover
import os

# LLM configuration
_TOKEN = os.getenv("ZELIBOBA_TOKEN", "").strip()
if not _TOKEN:
    raise RuntimeError("ZELIBOBA_TOKEN environment variable not set")

_ANSWER_URL = (
    "http://zeliboba.yandex-team.ru/balance/32b_aligned_quantized_202502/generative"
)
_EVAL_URL = (
    "http://zeliboba.yandex-team.ru/balance/qwen3_23B_edu_ml/v1/chat/completions"
)

_HEADERS = {
    "Content-Type": "application/json",
    "X-Model-Discovery-Oauth-Token": _TOKEN,
    "Authorization": "Bearer EMPTY",
}


def call_generation_llm(messages, max_tokens: int = 32000, temperature: int = 0) -> str:
    """Send messages to the answer LLM and return the text response."""
    payload = {
        "model": "does_not_matter",
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    print(f"[LLM Request] url={_ANSWER_URL} payload={json.dumps(payload)}")
    resp = requests.post(_ANSWER_URL, headers=_HEADERS, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_validation_llm(messages, max_tokens: int = 32000, temperature: int = 0) -> str:
    """Send messages to the validation LLM and return the text response."""
    payload = {
        "model": "does_not_matter",
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    print(f"[LLM Request] url={_EVAL_URL} payload={json.dumps(payload)}")
    resp = requests.post(_EVAL_URL, headers=_HEADERS, json=payload)
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


def evaluate_answer(answer: str) -> str:
    """Use the evaluation LLM to score an answer."""
    messages = [
        {
            "role": "system",
            "content": (
                "Return XML flags <correctness>Yes/No</correctness>"
                "<code_presence>Yes/No</code_presence>"
                "<line_reference>Yes/No</line_reference> only."
            ),
        },
        {"role": "user", "content": answer},
    ]
    return call_validation_llm(messages)


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
    async for session in get_session():
        total = 0
        counts = {
            "correctness": 0,
            "code_presence": 0,
            "line_reference": 0,
            "final": 0,
        }
        for case in BASKET:
            prompt_text = template.render(**case)
            # Log the rendered prompt for debugging/inspection
            print(f"[prompt:{prompt_id} case:{case['id']}] {prompt_text}")
            answer = call_generation_llm([{"role": "user", "content": prompt_text}])
            raw_eval = evaluate_answer(answer)
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
            counts["correctness"] += 1 if flags["correctness"] else 0
            counts["code_presence"] += 1 if flags["code_presence"] else 0
            counts["line_reference"] += 1 if flags["line_reference"] else 0
            counts["final"] += 1 if final_flag else 0
            total += 1
        await session.commit()
        if total:
            metrics = {
                "correctness": counts["correctness"] / total,
                "code_presence": counts["code_presence"] / total,
                "line_reference": counts["line_reference"] / total,
                "final_score": counts["final"] / total,
            }
            result = await session.exec(
                select(PromptVersion).where(PromptVersion.prompt_id == prompt_id)
            )
            pv = result.one()
            pv.metrics = metrics
            session.add(pv)
            await session.commit()
