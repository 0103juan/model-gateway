"""No test calls the API: a fake client records the requests it is sent."""

import asyncio
import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

from model_gateway import AsyncGateway, BudgetExceeded, Gateway, cost


class FakeMessages:
    """Stands in for client.beta.messages: every call uses 1,000 input and 100 output tokens."""

    def __init__(self, answers_as=None, fails=None):
        self.requests = []
        self.answers_as = answers_as  # the model that answers, when it is not the one that was asked
        self.fails = fails or {}  # model -> the error it raises

    def create(self, **request):
        self.requests.append(request)
        if request["model"] in self.fails:
            raise self.fails[request["model"]]
        return response(self.answers_as or request["model"])

    parse = create


def response(model: str, **usage):
    usage = SimpleNamespace(**{"input_tokens": 1000, "output_tokens": 100, **usage})
    return SimpleNamespace(model=model, stop_reason="end_turn", usage=usage)


def fake_client(**options):
    client = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(**options)), options={})
    client.with_options = lambda **options: client.options.update(options) or client
    return client


REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(code: int, kind=anthropic.APIStatusError):
    return kind("failed", response=httpx2.Response(code, request=REQUEST), body=None)


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


def test_a_key_that_has_spent_its_budget_gets_no_more_calls():
    client = fake_client()
    gateway = Gateway(client, key="eval", budget_usd=0.005)  # each call costs $0.003
    gateway.create(task="generate", messages=[])
    gateway.create(task="generate", messages=[])  # $0.003 spent is still under the budget, so this one runs

    with pytest.raises(BudgetExceeded, match="eval"):
        gateway.create(task="generate", messages=[])
    assert len(client.beta.messages.requests) == 2 and gateway.spent == pytest.approx(0.006)


def test_the_budget_of_a_key_holds_across_runs_and_does_not_touch_other_keys(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    Gateway(fake_client(), key="eval", ledger=ledger).create(task="generate", messages=[])

    with pytest.raises(BudgetExceeded):
        Gateway(fake_client(), key="eval", budget_usd=0.003, ledger=ledger).create(task="generate", messages=[])
    Gateway(fake_client(), key="demo", budget_usd=0.003, ledger=ledger).create(task="generate", messages=[])


@pytest.mark.parametrize("error", [
    status_error(429, anthropic.RateLimitError), status_error(500, anthropic.InternalServerError), status_error(529),
    anthropic.APITimeoutError(REQUEST), anthropic.APIConnectionError(request=REQUEST)],
    ids=["rate limit", "server error", "overloaded", "timeout", "no connection"])
def test_a_model_that_fails_hands_the_call_to_the_next_tier(error):
    client = fake_client(fails={"claude-haiku-4-5": error})
    gateway = Gateway(client, routes={"rewrite": ["small", "large"]})

    answer = gateway.create(task="rewrite", messages=[])

    assert answer.model == "claude-sonnet-5-5" and len(client.beta.messages.requests) == 2
    assert (gateway.calls[0]["tier"], gateway.calls[0]["fallback"]) == ("large", True)


def test_the_first_tier_answers_when_it_can():
    gateway = Gateway(fake_client(), routes={"rewrite": ["small", "large"]})
    gateway.create(task="rewrite", messages=[])
    assert (gateway.calls[0]["tier"], gateway.calls[0]["fallback"]) == ("small", False)


def test_a_rejected_request_is_raised_without_trying_another_model():
    client = fake_client(fails={"claude-haiku-4-5": status_error(400, anthropic.BadRequestError)})
    gateway = Gateway(client, routes={"rewrite": ["small", "large"]})

    with pytest.raises(anthropic.BadRequestError):
        gateway.create(task="rewrite", messages=[])
    assert len(client.beta.messages.requests) == 1 and gateway.calls == []


def test_when_every_tier_fails_the_last_error_is_raised_and_nothing_is_charged():
    client = fake_client(fails={"claude-haiku-4-5": status_error(529),
                                "claude-sonnet-5-5": anthropic.APITimeoutError(REQUEST)})
    gateway = Gateway(client, routes={"rewrite": ["small", "large"]})

    with pytest.raises(anthropic.APITimeoutError):
        gateway.create(task="rewrite", messages=[])
    assert gateway.calls == [] and gateway.spent == 0


def test_the_timeout_applies_to_each_attempt():
    client = fake_client()
    Gateway(client, timeout=30).create(task="generate", messages=[])
    assert client.options == {"timeout": 30}


def test_a_route_without_tiers_fails_at_construction():
    with pytest.raises(ValueError):
        Gateway(fake_client(), routes={"rewrite": []})


class Rewrite(BaseModel):
    standalone: str


def sdk_client(stop_reason="end_turn"):
    """The real SDK over a fake HTTP transport, so the cache stores and restores real response objects."""
    bodies = []

    def api(request):
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json={
            "id": f"msg_{len(bodies)}", "type": "message", "role": "assistant", "model": bodies[-1]["model"],
            "stop_reason": stop_reason, "stop_sequence": None, "usage": {"input_tokens": 1000, "output_tokens": 100},
            "content": [{"type": "text", "text": '{"standalone": "hotel cap"}'}]})

    http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(api))
    return anthropic.Anthropic(api_key="test", http_client=http), bodies


def ask(gateway, question="hotel cap?", **request):
    return gateway.parse(task="rewrite", messages=[{"role": "user", "content": question}],
                         output_format=Rewrite, **request)


def test_a_repeated_request_is_served_from_disk_at_no_cost(tmp_path):
    client, bodies = sdk_client()
    first = ask(Gateway(client, cache=tmp_path))
    gateway = Gateway(client, cache=tmp_path)  # a later run
    again = ask(gateway)

    assert len(bodies) == 1 and again.id == first.id
    assert again.parsed_output == first.parsed_output == Rewrite(standalone="hotel cap")
    assert gateway.calls[0]["cached"] and gateway.calls[0]["usd"] == 0 and gateway.calls[0]["input_tokens"] == 1000
    assert gateway.create(task="rewrite", messages=[{"role": "user", "content": "hotel cap?"}]).id == "msg_2"
    assert gateway.create(task="rewrite", messages=[{"role": "user", "content": "hotel cap?"}]).id == "msg_2"


def test_any_change_in_the_request_misses_the_cache(tmp_path):
    client, bodies = sdk_client()
    gateway = Gateway(client, routes={"cheap": "small"}, cache=tmp_path)
    ask(gateway)
    ask(gateway, question="hotel limit?")
    ask(gateway, system="Be brief.")
    gateway.parse(task="cheap", messages=[{"role": "user", "content": "hotel cap?"}], output_format=Rewrite)

    assert len(bodies) == 4 and not any(call["cached"] for call in gateway.calls)


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_an_unfinished_response_is_not_stored(tmp_path, stop_reason):
    client, bodies = sdk_client(stop_reason)
    gateway = Gateway(client, cache=tmp_path)
    gateway.create(task="generate", messages=[{"role": "user", "content": "hi"}])
    gateway.create(task="generate", messages=[{"role": "user", "content": "hi"}])

    assert len(bodies) == 2 and list(tmp_path.iterdir()) == []


def agent_api(fail=()):
    """A fake HTTP API for a two-turn agent: it asks for the tool `add`, then answers with what the tool returned.
    A model named in `fail` answers 500."""
    bodies = []

    def api(request):
        body = json.loads(request.content)
        bodies.append({**body, "beta": request.headers.get("anthropic-beta", "")})
        if body["model"] in fail:
            return httpx2.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "down"}})
        last = body["messages"][-1]["content"]
        results = [block["content"] for block in last if block["type"] == "tool_result"] if isinstance(last, list) else []
        content = ([{"type": "text", "text": f"It is {results[0]}."}] if results else
                   [{"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"a": 2, "b": 3}}])
        return httpx2.Response(200, json={
            "id": f"msg_{len(bodies)}", "type": "message", "role": "assistant", "model": body["model"],
            "stop_reason": "end_turn" if results else "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 100}, "content": content})

    return httpx2.MockTransport(api), bodies


@anthropic.beta_tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


@anthropic.beta_async_tool(name="add")
async def add_later(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


QUESTION = [{"role": "user", "content": "2 + 3?"}]


def test_every_turn_of_the_tool_runner_goes_through_the_gateway(tmp_path):
    transport, bodies = agent_api()
    client = anthropic.Anthropic(api_key="test", http_client=anthropic.DefaultHttpxClient(transport=transport))
    gateway = Gateway(client, routes={"agent": "small"}, cache=tmp_path)
    final = gateway.tool_runner(task="agent", messages=QUESTION, tools=[add]).until_done()

    assert final.content[0].text == "It is 5."
    assert [body["model"] for body in bodies] == ["claude-haiku-4-5"] * 2
    assert [call["task"] for call in gateway.calls] == ["agent", "agent"] and gateway.spent == pytest.approx(0.003)

    again = Gateway(client, routes={"agent": "small"}, cache=tmp_path)  # a later run of the same conversation
    assert again.tool_runner(task="agent", messages=QUESTION, tools=[add]).until_done().id == final.id
    assert len(bodies) == 2 and again.spent == 0


def test_a_turn_of_the_tool_runner_falls_back_with_the_settings_of_the_tier_that_runs_it():
    transport, bodies = agent_api(fail={"claude-haiku-4-5"})
    client = anthropic.Anthropic(api_key="test", max_retries=0,
                                 http_client=anthropic.DefaultHttpxClient(transport=transport))
    gateway = Gateway(client, routes={"agent": ["small", "large"]})
    gateway.tool_runner(task="agent", messages=QUESTION, tools=[add], max_tokens=500).until_done()

    large = [body for body in bodies if body["model"] == "claude-sonnet-5-5"]
    small = [body for body in bodies if body["model"] == "claude-haiku-4-5"]
    assert len(small) == len(large) == 2 and all(call["fallback"] for call in gateway.calls)
    assert all(body["max_tokens"] == 500 and body["fallbacks"] == "default" and "server-side-fallback" in body["beta"]
               for body in large)
    assert not any("fallbacks" in body or "server-side-fallback" in body["beta"] for body in small)


def test_the_tool_runner_does_not_stream():
    with pytest.raises(ValueError):
        Gateway(fake_client()).tool_runner(task="agent", messages=QUESTION, tools=[add], stream=True)


def test_the_async_gateway_awaits_calls_and_runs_the_async_tool_runner():
    transport, bodies = agent_api(fail={"claude-haiku-4-5"})
    client = anthropic.AsyncAnthropic(api_key="test", max_retries=0,
                                      http_client=anthropic.DefaultAsyncHttpxClient(transport=transport))
    gateway = AsyncGateway(client, routes={"agent": ["small", "large"]}, budget_usd=0.005)

    async def run():
        final = await gateway.tool_runner(task="agent", messages=QUESTION, tools=[add_later]).until_done()
        with pytest.raises(BudgetExceeded):
            await gateway.create(task="agent", messages=QUESTION)
        return final

    assert asyncio.run(run()).content[0].text == "It is 5."
    assert [call["model"] for call in gateway.calls] == ["claude-sonnet-5-5"] * 2 and gateway.calls[0]["fallback"]
