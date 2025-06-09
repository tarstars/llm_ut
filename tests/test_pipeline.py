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


def test_evaluate_answer_defaults_to_generation_llm(monkeypatch):
    import pipeline

    captured = {}

    def fake_call(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return (
            "<correctness>Yes</correctness>"
            "<code_presence>No</code_presence>"
            "<line_reference>No</line_reference>"
        )

    monkeypatch.setattr(pipeline, "call_generation_llm", fake_call)

    result = pipeline.evaluate_answer("some answer")

    assert "<correctness>" in result
    assert captured["messages"][1]["content"] == "some answer"


def test_evaluate_answer_custom_llm(monkeypatch):
    import pipeline

    captured = {}

    def custom_call(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return (
            "<correctness>No</correctness>"
            "<code_presence>No</code_presence>"
            "<line_reference>Yes</line_reference>"
        )

    result = pipeline.evaluate_answer("foo", llm_fn=custom_call, extra=1)

    assert "<correctness>" in result
    assert captured["messages"][1]["content"] == "foo"
    assert captured["kwargs"] == {"extra": 1}
