import json
import asyncio
import sys
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util import Retry
except ModuleNotFoundError:  # allow running tests without dependency
    class Retry:
        def __init__(self, *args, **kwargs):
            pass

    class HTTPAdapter:
        def __init__(self, *args, **kwargs):
            pass

    class _FakeSession:
        def mount(self, *args, **kwargs):
            pass

        def post(self, *args, **kwargs):
            raise RuntimeError("requests library not installed")

    class _FakeExceptions:
        class RequestException(Exception):
            pass

        class ConnectionError(RequestException):
            pass

    class requests:  # noqa: D401 - simple stub
        """Fallback requests-like interface for tests without dependency."""

        Session = _FakeSession
        exceptions = _FakeExceptions
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
import time
try:
    from sqlmodel import select
    from app.model import PromptVersion, ResponseRecord, LLMInteraction
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

_EVAL_TOKEN = os.getenv("ELIZA_API", "").strip()
if not _EVAL_TOKEN:
    raise RuntimeError("ELIZA_API environment variable not set")

_ANSWER_URL = (
    "http://zeliboba.yandex-team.ru/balance/32b_aligned_quantized_202502/generative"
)
_EVAL_URL_DEFAULT = (
    "https://api.eliza.yandex.net/together/v1/chat/completions"
)
_USE_SINGLE_LLM_ENDPOINT = os.getenv("USE_SINGLE_LLM_ENDPOINT", "false").lower() in (
    "1",
    "true",
    "yes",
)
_EVAL_URL = _ANSWER_URL if _USE_SINGLE_LLM_ENDPOINT else _EVAL_URL_DEFAULT

_HEADERS = {
    "Content-Type": "application/json",
    "X-Model-Discovery-Oauth-Token": _TOKEN,
}

_EVAL_HEADERS = {
    "Content-Type": "application/json",
    "authorization": f"OAuth {_EVAL_TOKEN}",
}

_SESSION = requests.Session()
_SESSION.mount("http://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.5)))
_SESSION.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.5)))


def call_generation_llm(messages, params=None) -> str:
    """Send messages to the answer LLM and return the text response."""
    if params is None:
        params = {"NumHypos": 1, "Seed": 42}
    payload = {
        "messages": messages,
        "Params": params,
    }
    print(
        f"[LLM Request] url={_ANSWER_URL} payload={json.dumps(payload, ensure_ascii=False)}"
    )
    for attempt in range(3):
        try:
            resp = _SESSION.post(_ANSWER_URL, headers=_HEADERS, json=payload)
            resp.raise_for_status()
            data = resp.json()
            chunk = data["Responses"][0]
            if not chunk.get("ReachedEos"):
                raise RuntimeError("generation did not finish with eos")
            return chunk["Response"]
        except requests.exceptions.RequestException as exc:
            if attempt == 2:
                raise RuntimeError("failed to call generation LLM") from exc
            time.sleep(2 ** attempt)


def call_validation_llm(messages, max_tokens: int = 32000, temperature: int = 0) -> str:
    """Send messages to the validation LLM and return the text response.

    The service may return different JSON envelopes depending on the
    deployment.  Historically the response mimicked the generation API and
    contained a ``Responses`` array.  Newer versions wrap the model output in a
    ``response`` object similar to the OpenAI format.  This helper extracts the
    assistant message from either structure.
    """

    payload = {
        "model": "deepseek-ai/deepseek-v3",
        "messages": messages,
    }
    print(
        f"[LLM Request] url={_EVAL_URL} payload={json.dumps(payload, ensure_ascii=False)}"
    )
    for attempt in range(3):
        try:
            resp = _SESSION.post(_EVAL_URL, headers=_EVAL_HEADERS, json=payload)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.RequestException as exc:
            if attempt == 2:
                raise RuntimeError("failed to call validation LLM") from exc
            time.sleep(2 ** attempt)

    # Legacy format: {"Responses": [{"Response": "...", "ReachedEos": True}]}
    if "Responses" in data:
        chunk = data["Responses"][0]
        if not chunk.get("ReachedEos"):
            raise RuntimeError("validation generation did not finish with eos")
        return chunk["Response"]

    # New format may include an outer "response" envelope as shown in the
    # evaluation example.  Extract the content from the first choice.
    container_obj = data.get("response", data)
    try:
        return container_obj["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001 - handle unexpected responses
        raise RuntimeError("unexpected validation response format") from exc


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
    return_messages: bool = False,
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
    result = llm_fn(messages, **llm_kwargs)
    return (result, messages) if return_messages else result


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


async def log_interaction(
    session: AsyncSession,
    prompt_id: str,
    question_id: int,
    request_payload: Dict,
    response: str,
    request_type: str,
) -> None:
    """Persist a request/response pair to the database."""
    rec = LLMInteraction(
        prompt_id=prompt_id,
        question_id=int(question_id),
        request_type=request_type,
        request_payload=request_payload,
        response=response,
    )
    session.add(rec)


async def process_prompt(prompt_id: str, text: str, eval_fn=None, **eval_kwargs):
    """Evaluate a prompt concurrently using up to 5 parallel LLM calls."""
    template = Template(text)
    logger.info(
        "[process_prompt] start prompt_id=%s cases=%d", prompt_id, len(BASKET)
    )

    # Acquire a database session early so missing SQL drivers raise before
    # launching any network requests.
    async for _ in get_session():
        break

    semaphore = asyncio.Semaphore(5)

    async def handle_case(case):
        async with semaphore:
            prompt_text = template.render(**case)
            messages = [{"role": "user", "content": prompt_text}]
            # Log the rendered prompt for debugging/inspection
            print(f"[prompt:{prompt_id} case:{case['id']}] {prompt_text}")

            try:
                answer = await asyncio.to_thread(call_generation_llm, messages)
                raw_eval, eval_msgs = await asyncio.to_thread(
                    evaluate_answer,
                    case.get("program", ""),
                    case.get("error", ""),
                    answer,
                    llm_fn=eval_fn,
                    return_messages=True,
                    **eval_kwargs,
                )
            except RuntimeError as exc:
                logger.error(
                    "[process_prompt] case %s failed: %s", case["id"], exc
                )
                flags = {
                    "right_error_description": False,
                    "correct_hint": False,
                    "correct_or_absent_code": False,
                    "correct_line_reference": False,
                }
                return flags, False

            flags = parse_flags(raw_eval)
            final_flag = all(flags.values())

            async for session in get_session():
                await log_interaction(
                    session,
                    prompt_id,
                    case["id"],
                    {"messages": messages},
                    answer,
                    "generation",
                )
                await log_interaction(
                    session,
                    prompt_id,
                    case["id"],
                    {"messages": eval_msgs},
                    raw_eval,
                    "validation",
                )
                rec = ResponseRecord(
                    prompt_id=prompt_id,
                    case_id=case["id"],
                    answer=answer,
                    raw_evaluation=raw_eval,
                    flags=flags,
                    final_flag=final_flag,
                )
                session.add(rec)
                await session.commit()

            logger.info(
                "[process_prompt] case=%s flags=%s final=%s",
                case["id"],
                flags,
                final_flag,
            )

            return flags, final_flag

    tasks = [asyncio.create_task(handle_case(case)) for case in BASKET]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results = []
    for item in raw_results:
        if isinstance(item, Exception):
            logger.error("[process_prompt] unexpected error: %s", item)
        else:
            results.append(item)

    total = len(results)
    counts = {
        "right_error_description": 0,
        "correct_hint": 0,
        "correct_or_absent_code": 0,
        "correct_line_reference": 0,
        "final": 0,
    }

    for flags, final_flag in results:
        counts["right_error_description"] += 1 if flags["right_error_description"] else 0
        counts["correct_hint"] += 1 if flags["correct_hint"] else 0
        counts["correct_or_absent_code"] += 1 if flags["correct_or_absent_code"] else 0
        counts["correct_line_reference"] += 1 if flags["correct_line_reference"] else 0
        counts["final"] += 1 if final_flag else 0

    if total:
        metrics = {
            "right_error_description": counts["right_error_description"] / total,
            "correct_hint": counts["correct_hint"] / total,
            "correct_or_absent_code": counts["correct_or_absent_code"] / total,
            "correct_line_reference": counts["correct_line_reference"] / total,
            "final_score": counts["final"] / total,
        }

        async for session in get_session():
            result = await session.exec(
                select(PromptVersion).where(PromptVersion.prompt_id == prompt_id)
            )
            pv = result.one()
            pv.metrics = metrics
            session.add(pv)
            await session.commit()
            logger.info("[process_prompt] final metrics for %s: %s", prompt_id, metrics)
