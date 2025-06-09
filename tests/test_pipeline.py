import os
import sys
import asyncio
import pytest

# Ensure the project root is on the Python path so "pipeline" can be imported
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault('ZELIBOBA_TOKEN', 'dummy')

def test_process_prompt_triggers_missing_greenlet_error():
    from pipeline import process_prompt

    with pytest.raises(Exception) as excinfo:
        asyncio.run(process_prompt('p1', '{{ value }}'))

    assert 'greenlet_spawn' in str(excinfo.value)


def test_call_generation_llm_parses_response(monkeypatch):
    import pipeline

    captured = {}

    def fake_post(url, headers=None, json=None):
        captured['url'] = url
        captured['headers'] = headers
        captured['json'] = json

        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "Responses": [
                        {"Response": "hello", "ReachedEos": True}
                    ]
                }

        return Resp()

    monkeypatch.setattr(pipeline.requests, "post", fake_post)

    params = {"NumHypos": 1, "Seed": 42}
    result = pipeline.call_generation_llm(
        [{"role": "user", "content": "hi"}], params=params
    )

    assert result == "hello"
    assert captured['url'] == pipeline._ANSWER_URL
    assert captured['headers'] == pipeline._HEADERS
    assert captured['json'] == {
        "Params": params,
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_call_generation_llm_raises_on_no_eos(monkeypatch):
    import pipeline

    def fake_post(url, headers=None, json=None):
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "Responses": [
                        {"Response": "oops", "ReachedEos": False}
                    ]
                }

        return Resp()

    monkeypatch.setattr(pipeline.requests, "post", fake_post)

    with pytest.raises(RuntimeError):
        pipeline.call_generation_llm([{"role": "user", "content": "hi"}])


def test_call_validation_llm_single_endpoint(monkeypatch):
    import pipeline

    captured = {}

    def fake_post(url, headers=None, json=None):
        captured['url'] = url
        captured['headers'] = headers
        captured['json'] = json

        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "Responses": [
                        {"Response": "ok", "ReachedEos": True}
                    ]
                }

        return Resp()

    monkeypatch.setattr(pipeline.requests, "post", fake_post)
    monkeypatch.setattr(pipeline, "_USE_SINGLE_LLM_ENDPOINT", True)
    monkeypatch.setattr(pipeline, "_EVAL_URL", "http://example.com/eval")

    result = pipeline.call_validation_llm([{"role": "user", "content": "hi"}])

    assert result == "ok"
    assert captured['url'] == pipeline._EVAL_URL
    assert captured['headers'] == pipeline._HEADERS
    assert captured['json'] == {
        "messages": [{"role": "user", "content": "hi"}],
        "Params": {"NumHypos": 1, "Seed": 42},
    }


def test_call_validation_llm_separate_endpoint(monkeypatch):
    import pipeline

    captured = {}

    def fake_post(url, headers=None, json=None):
        captured['url'] = url
        captured['headers'] = headers
        captured['json'] = json

        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "Responses": [
                        {"Response": "fine", "ReachedEos": True}
                    ]
                }

        return Resp()

    monkeypatch.setattr(pipeline.requests, "post", fake_post)
    monkeypatch.setattr(pipeline, "_USE_SINGLE_LLM_ENDPOINT", False)
    monkeypatch.setattr(pipeline, "_EVAL_URL", "http://other.com/eval")

    result = pipeline.call_validation_llm(
        [{"role": "user", "content": "hi"}], max_tokens=10, temperature=1
    )

    assert result == "fine"
    assert captured['url'] == pipeline._EVAL_URL
    assert captured['headers'] == pipeline._HEADERS
    assert captured['json'] == {
        "messages": [{"role": "user", "content": "hi"}],
        "Params": {"NumHypos": 1, "Seed": 42},
    }


def test_evaluate_answer_sends_user_message(monkeypatch):
    import pipeline

    captured = {}

    def fake_llm(messages, **kwargs):
        captured['messages'] = messages
        return "<right_error_description>Yes</right_error_description>"

    program = "print(1)"
    error = ""
    advice = "do nothing"

    pipeline.evaluate_answer(program, error, advice, llm_fn=fake_llm)

    assert captured['messages'][0]['role'] == 'user'
    assert program in captured['messages'][0]['content']
