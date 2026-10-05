"""No test calls the API: a fake client records the requests it is sent."""

from types import SimpleNamespace

import pytest

from model_gateway import Gateway


class FakeMessages:
    """Stands in for client.beta.messages."""

    def __init__(self):
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        return SimpleNamespace(model=request["model"], stop_reason="end_turn")

    parse = create


def fake_client():
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages()))


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
