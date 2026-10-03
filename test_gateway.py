"""No test calls the API: a fake client records the requests it is sent."""

import json
from types import SimpleNamespace

import pytest

from model_gateway import Gateway, cost


class FakeMessages:
    """Stands in for client.beta.messages: every call uses 1,000 input and 100 output tokens."""

    def __init__(self, answers_as=None):
        self.requests = []
        self.answers_as = answers_as  # the model that answers, when it is not the one that was asked

    def create(self, **request):
        self.requests.append(request)
        return response(self.answers_as or request["model"])

    parse = create


def response(model: str, **usage):
    usage = SimpleNamespace(**{"input_tokens": 1000, "output_tokens": 100, **usage})
    return SimpleNamespace(model=model, stop_reason="end_turn", usage=usage)


def fake_client(**options):
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(**options)))


def sent(client) -> dict:
    return client.beta.messages.requests[-1]


def test_a_task_runs_on_the_tier_its_route_names():
    client = fake_client()
    Gateway(client, routes={"rewrite": "small"}).create(task="rewrite", messages=[])
    assert sent(client)["model"] == "claude-haiku-4-5"


def test_a_task_without_a_route_runs_on_the_large_model():
    client = fake_client()
    Gateway(client, routes={"rewrite": "small"}).parse(task="judge", messages=[])
    assert sent(client)["model"] == "claude-sonnet-5-5" and sent(client)["fallbacks"] == "default"


def test_effort_is_removed_only_for_a_model_that_rejects_it():
    client = fake_client()
    gateway = Gateway(client, routes={"rewrite": "small"})
    schema = {"type": "json_schema", "schema": {}}

    gateway.create(task="rewrite", messages=[], output_config={"effort": "low"})
    assert "output_config" not in sent(client)
    gateway.create(task="rewrite", messages=[], output_config={"effort": "low", "format": schema})
    assert sent(client)["output_config"] == {"format": schema}
    gateway.create(task="generate", messages=[], output_config={"effort": "low"})
    assert sent(client)["output_config"] == {"effort": "low"}


def test_the_callers_settings_win_over_the_tiers():
    client = fake_client()
    Gateway(client).create(task="grade", messages=[], max_tokens=256)
    assert sent(client)["max_tokens"] == 256


def test_the_response_is_the_one_the_sdk_returned():
    client = fake_client()
    response = Gateway(client).create(task="generate", messages=[{"role": "user", "content": "hi"}])
    assert response.model == "claude-sonnet-5-5" and sent(client)["messages"][0]["content"] == "hi"


def test_a_route_to_a_tier_that_does_not_exist_fails_at_construction():
    with pytest.raises(ValueError, match="medium"):
        Gateway(fake_client(), routes={"rewrite": "medium"})


def test_a_call_is_recorded_with_its_cost_at_list_price():
    gateway = Gateway(fake_client(), routes={"rewrite": "small"}, key="eval")
    gateway.create(task="rewrite", messages=[])
    gateway.create(task="generate", messages=[])

    small, large = gateway.calls
    assert (small["key"], small["task"], small["tier"]) == ("eval", "rewrite", "small")
    assert small["model"] == "claude-haiku-4-5"
    assert small["usd"] == pytest.approx((1000 * 1.00 + 100 * 5.00) / 1e6)
    assert large["usd"] == pytest.approx((1000 * 2.00 + 100 * 10.00) / 1e6)
    assert (large["input_tokens"], large["output_tokens"]) == (1000, 100)


def test_the_price_is_that_of_the_model_that_answered():
    gateway = Gateway(fake_client(answers_as="claude-opus-4-8"))
    gateway.create(task="generate", messages=[])
    assert gateway.calls[0]["model"] == "claude-opus-4-8"
    assert gateway.calls[0]["usd"] == pytest.approx((1000 * 5.00 + 100 * 25.00) / 1e6)


def test_a_price_covers_snapshots_and_the_longest_name_wins():
    assert cost(response("claude-haiku-4-5-20251001"))["usd"] == pytest.approx(0.0015)
    assert cost(response("claude-opus-5-5"))["usd"] == pytest.approx((1000 * 4.00 + 100 * 20.00) / 1e6)
    assert cost(response("claude-opus-5"))["usd"] == pytest.approx((1000 * 5.00 + 100 * 25.00) / 1e6)


def test_a_model_with_no_listed_price_is_charged_at_the_highest():
    assert cost(response("some-future-model"))["usd"] == pytest.approx((1000 * 10.00 + 100 * 50.00) / 1e6)


def test_prompt_cache_tokens_count_as_input():
    cached = response("claude-sonnet-5-5", cache_creation_input_tokens=500, cache_read_input_tokens=None)
    assert cost(cached)["input_tokens"] == 1500


def test_the_ledger_gets_one_line_per_call(tmp_path):
    ledger = tmp_path / "runs" / "ledger.jsonl"
    gateway = Gateway(fake_client(), key="eval", ledger=ledger)
    gateway.create(task="generate", messages=[])
    gateway.parse(task="judge", messages=[])

    lines = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert lines == gateway.calls and [line["task"] for line in lines] == ["generate", "judge"]
