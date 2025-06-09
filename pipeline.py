import json
import asyncio
import sys
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
import logging
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

# Ensure console output uses UTF-8 encoding when printing logs
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# LLM configuration
_TOKEN = os.getenv("ZELIBOBA_TOKEN", "").strip()
if not _TOKEN:
    raise RuntimeError("ZELIBOBA_TOKEN environment variable not set")

_ANSWER_URL = (
    "http://zeliboba.yandex-team.ru/balance/32b_aligned_quantized_202502/generative"
)
_EVAL_URL_DEFAULT = (
    "http://zeliboba.yandex-team.ru/balance/qwen3_23B_edu_ml/v1/chat/completions"
)
_USE_SINGLE_LLM_ENDPOINT = os.getenv("USE_SINGLE_LLM_ENDPOINT", "true").lower() in (
    "1",
    "true",
    "yes",
)
_EVAL_URL = _ANSWER_URL if _USE_SINGLE_LLM_ENDPOINT else _EVAL_URL_DEFAULT

_HEADERS = {
    "Content-Type": "application/json",
    "X-Model-Discovery-Oauth-Token": _TOKEN,
}


def call_generation_llm(messages, params=None) -> str:
    """Send messages to the answer LLM and return the text response."""
    if params is None:
        params = {"NumHypos": 1, "Seed": 42}
    payload = {
        "messages": messages,
        "Params": params,
    }
    print(f"[LLM Request] url={_ANSWER_URL} payload={json.dumps(payload, ensure_ascii=False)}")
    resp = requests.post(_ANSWER_URL, headers=_HEADERS, json=payload)
    resp.raise_for_status()
    data = resp.json()
    chunk = data["Responses"][0]
    if not chunk.get("ReachedEos"):
        raise RuntimeError("generation did not finish with eos")
    return chunk["Response"]


def call_validation_llm(messages, max_tokens: int = 32000, temperature: int = 0) -> str:
    """Send messages to the validation LLM and return the text response."""
    if _USE_SINGLE_LLM_ENDPOINT:
        payload = {
            "messages": messages,
            "Params": {"NumHypos": 1, "Seed": 42},
        }
        print(f"[LLM Request] url={_EVAL_URL} payload={json.dumps(payload, ensure_ascii=False)}")
        resp = requests.post(_EVAL_URL, headers=_HEADERS, json=payload)
        resp.raise_for_status()
        data = resp.json()
        chunk = data["Responses"][0]
        if not chunk.get("ReachedEos"):
            raise RuntimeError("validation generation did not finish with eos")
        return chunk["Response"]
    else:
        payload = {
            "model": "does_not_matter",
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        print(f"[LLM Request] url={_EVAL_URL} payload={json.dumps(payload, ensure_ascii=False)}")
        resp = requests.post(_EVAL_URL, headers=_HEADERS, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


with open("basket.json") as f:
    BASKET = json.load(f)


def simple_evaluate(answer: str) -> str:
    right_error_description = "Yes" if "error" in answer else "No"
    correct_hint = "Yes" if "fix" in answer else "No"
    correct_or_absent_code = "Yes" if "def" not in answer else "No"
    correct_line_reference = "Yes" if "line" in answer else "No"
    return (
        f"<right_error_description>{right_error_description}</right_error_description>"
        f"<correct_hint>{correct_hint}</correct_hint>"
        f"<correct_or_absent_code>{correct_or_absent_code}</correct_or_absent_code>"
        f"<correct_line_reference>{correct_line_reference}</correct_line_reference>"
    )


def evaluate_answer(
    program: str,
    error: str,
    advice: str,
    llm_fn=None,
    **llm_kwargs,
) -> str:
    """Use a configurable LLM function to score an answer."""
    if llm_fn is None:
        llm_fn = call_validation_llm

    prompt = (
        "You are an assessor of debugging advice. "
        "Here is the student's program:\n{program}\n\n"
        "The following text is the error:\n{error}\n\n"
        "The tutor suggested this advice:\n{advice}\n\n"
        "Evaluate the advice according to these criteria:\n"
        "1. Right error description\n"
        "2. Correct hint\n"
        "3. Correct or absent code – either the tutor gives no code or the code shown is correct. Reply 'No' only if the tutor's code snippet contains mistakes.\n"
        "4. Correct line reference\n\n"
        "Return only the XML tags for each criterion:"
        " <right_error_description>Yes/No</right_error_description>"
        " <correct_hint>Yes/No</correct_hint>"
        " <correct_or_absent_code>Yes/No</correct_or_absent_code>"
        " <correct_line_reference>Yes/No</correct_line_reference>."
    ).format(program=program, error=error, advice=advice)

    messages = [
        {"role": "user", "content": prompt},
    ]
    return llm_fn(messages, **llm_kwargs)


def parse_flags(raw: str) -> Dict[str, bool]:
    def extract(tag: str) -> bool:
        open_t = f"<{tag}>"
        close_t = f"</{tag}>"
        if open_t in raw and close_t in raw:
            val = raw.split(open_t)[1].split(close_t)[0].strip()
            return val.lower() == "yes"
        return False

    return {
        "right_error_description": extract("right_error_description"),
        "correct_hint": extract("correct_hint"),
        "correct_or_absent_code": extract("correct_or_absent_code"),
        "correct_line_reference": extract("correct_line_reference"),
    }


async def process_prompt(prompt_id: str, text: str, eval_fn=None, **eval_kwargs):
    template = Template(text)
    logger.info("[process_prompt] start prompt_id=%s cases=%d", prompt_id, len(BASKET))
    async for session in get_session():
        total = 0
        counts = {
            "right_error_description": 0,
            "correct_hint": 0,
            "correct_or_absent_code": 0,
            "correct_line_reference": 0,
            "final": 0,
        }
        for case in BASKET:
            prompt_text = template.render(**case)
            # Log the rendered prompt for debugging/inspection
            print(f"[prompt:{prompt_id} case:{case['id']}] {prompt_text}")
            answer = call_generation_llm([{"role": "user", "content": prompt_text}])
            logger.info("[process_prompt] case=%s answer=%s", case["id"], answer)
            raw_eval = evaluate_answer(
                case.get("program", ""),
                case.get("error", ""),
                answer,
                llm_fn=eval_fn,
                **eval_kwargs,
            )
            logger.info("[process_prompt] case=%s evaluation=%s", case["id"], raw_eval)
            flags = parse_flags(raw_eval)
            final_flag = all(flags.values())
            logger.info("[process_prompt] case=%s flags=%s final=%s", case["id"], flags, final_flag)
            rec = ResponseRecord(
                prompt_id=prompt_id,
                case_id=case["id"],
                answer=answer,
                raw_evaluation=raw_eval,
                flags=flags,
                final_flag=final_flag,
            )
            session.add(rec)
            counts["right_error_description"] += 1 if flags["right_error_description"] else 0
            counts["correct_hint"] += 1 if flags["correct_hint"] else 0
            counts["correct_or_absent_code"] += 1 if flags["correct_or_absent_code"] else 0
            counts["correct_line_reference"] += 1 if flags["correct_line_reference"] else 0
            counts["final"] += 1 if final_flag else 0
            total += 1
            logger.info("[process_prompt] running counts=%s total=%d", counts, total)
        await session.commit()
        if total:
            metrics = {
                "right_error_description": counts["right_error_description"] / total,
                "correct_hint": counts["correct_hint"] / total,
                "correct_or_absent_code": counts["correct_or_absent_code"] / total,
                "correct_line_reference": counts["correct_line_reference"] / total,
                "final_score": counts["final"] / total,
            }
            result = await session.exec(
                select(PromptVersion).where(PromptVersion.prompt_id == prompt_id)
            )
            pv = result.one()
            pv.metrics = metrics
            session.add(pv)
            await session.commit()
            logger.info("[process_prompt] final metrics for %s: %s", prompt_id, metrics)
