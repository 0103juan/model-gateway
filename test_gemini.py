"""GeminiClient with a fake google-genai client: no test calls Vertex AI."""

from types import SimpleNamespace

import pytest
from google.genai import errors, types
from pydantic import BaseModel

from model_gateway import GEMINI_TIERS, Gateway, GeminiClient


class Grade(BaseModel):
    correct: bool


class FakeModels:
    def __init__(self, text="hola", finish="STOP", fails=None):
        self.calls, self.text, self.finish, self.fails = [], text, finish, fails or {}

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if model in self.fails:
            raise self.fails[model]
        return SimpleNamespace(
            text=self.text, response_id="r1", model_version=model,
            candidates=[SimpleNamespace(finish_reason=types.FinishReason[self.finish])],
            usage_metadata=SimpleNamespace(prompt_token_count=1000, candidates_token_count=100, thoughts_token_count=300))


def gemini(**options):
    models = FakeModels(**options)
    return GeminiClient(client=SimpleNamespace(models=models)), models


def test_the_request_is_translated_and_thinking_counts_as_output():
    client, models = gemini()
    response = Gateway(client, tiers=GEMINI_TIERS, routes={"answer": "small"}).create(
        task="answer", system="Eres breve.", output_config={"effort": "low"},
        messages=[{"role": "user", "content": "hola"}, {"role": "assistant", "content": [{"type": "text", "text": "¿sí?"}]}])

    call = models.calls[-1]
    assert call["model"] == "gemini-3.5-flash-lite"
    assert call["config"].system_instruction == "Eres breve." and call["config"].max_output_tokens == 16000
    assert call["config"].thinking_config.thinking_level == types.ThinkingLevel.LOW
    assert call["config"].http_options.retry_options.attempts == 3  # a 503 is retried before falling back
    assert [content.role for content in call["contents"]] == ["user", "model"]
    assert response.content[0].text == "hola" and response.stop_reason == "end_turn"
    assert response.usage.output_tokens == 400  # 100 answer + 300 thinking


def test_cost_uses_the_lite_price_not_the_flash_one():
    client, _ = gemini()
    gateway = Gateway(client, tiers=GEMINI_TIERS, routes={"answer": "small"})
    gateway.create(task="answer", messages=[{"role": "user", "content": "hola"}])
    assert gateway.calls[-1]["usd"] == pytest.approx((1000 * 0.30 + 400 * 2.50) / 1e6)


def test_parse_returns_the_model_and_none_when_the_answer_does_not_fit():
    client, models = gemini(text='{"correct": true}')
    gateway = Gateway(client, tiers=GEMINI_TIERS)
    assert gateway.parse(task="grade", output_format=Grade, messages=[{"role": "user", "content": "x"}]).parsed_output == Grade(correct=True)
    assert models.calls[-1]["config"].response_mime_type == "application/json"

    cut, _ = gemini(text='{"corr', finish="MAX_TOKENS")
    response = Gateway(cut, tiers=GEMINI_TIERS).parse(task="grade", output_format=Grade, messages=[{"role": "user", "content": "x"}])
    assert response.parsed_output is None and response.stop_reason == "max_tokens"


@pytest.mark.parametrize("code, status", [(429, "RESOURCE_EXHAUSTED"), (499, "CANCELLED")])
def test_a_rate_limit_falls_back_to_the_next_tier_and_a_bad_request_does_not(code, status):
    limited = errors.ClientError(code, {"error": {"code": code, "message": "quota", "status": status}})
    client, models = gemini(fails={"gemini-3.5-flash-lite": limited})
    gateway = Gateway(client, tiers=GEMINI_TIERS, routes={"answer": ["small", "large"]})
    gateway.create(task="answer", messages=[{"role": "user", "content": "hola"}])
    assert [call["model"] for call in models.calls] == ["gemini-3.5-flash-lite", "gemini-3.5-flash"]
    assert gateway.calls[-1]["fallback"]

    bad = errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})
    client, models = gemini(fails={"gemini-3.5-flash-lite": bad})
    with pytest.raises(errors.ClientError):
        Gateway(client, tiers=GEMINI_TIERS, routes={"answer": ["small", "large"]}).create(
            task="answer", messages=[{"role": "user", "content": "hola"}])
    assert len(models.calls) == 1


def test_a_cached_structured_answer_comes_back_parsed_without_a_call(tmp_path):
    client, models = gemini(text='{"correct": false}')
    gateway = Gateway(client, tiers=GEMINI_TIERS, cache=tmp_path)
    request = {"task": "grade", "output_format": Grade, "messages": [{"role": "user", "content": "x"}]}
    gateway.parse(**request)
    again = gateway.parse(**request)
    assert len(models.calls) == 1 and again.parsed_output == Grade(correct=False) and gateway.calls[-1]["cached"]


def test_tools_are_refused_rather_than_dropped():
    client, _ = gemini()
    with pytest.raises(ValueError):
        Gateway(client, tiers=GEMINI_TIERS).create(task="x", tools=[{"name": "t"}], messages=[{"role": "user", "content": "x"}])
