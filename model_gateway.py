"""One entry point for model calls. The caller names the task; a table decides which model runs it and which
one takes over if it fails, a request already answered is served from disk, every call leaves a record of
what it cost, and a key that has spent its budget gets no more calls.

    gateway = Gateway(anthropic.Anthropic(), routes={"rewrite": ["small", "large"]}, key="eval", budget_usd=1.00,
                      ledger="ledger.jsonl", cache=".gateway/cache", timeout=60)
    response = gateway.parse(task="rewrite", system=..., messages=[...], output_format=Rewrite)
    gateway.calls[-1]["usd"]
"""

import hashlib
import json
import time
from pathlib import Path

import anthropic
from anthropic.types.beta import BetaMessage
from anthropic.types.beta.parsed_beta_message import ParsedBetaMessage
from pydantic import TypeAdapter

# A tier is a model plus the request settings that model needs.
# A safety decline on the large model is re-run server-side on Anthropic's recommended fallback model.
TIERS = {
    "small": {"model": "claude-haiku-4-5", "max_tokens": 16000},
    "large": {"model": "claude-sonnet-5-5", "max_tokens": 16000,
              "betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"},
}
NO_EFFORT = {"claude-haiku-4-5"}  # models that reject output_config.effort

# USD per million tokens (input, output), list price on 25 September 2026. A name also covers its dated
# snapshots and point releases ("claude-haiku-4-5-20251001", "claude-fable-5-1"); the longest match wins.
USD_PER_MTOK = {
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
UNFINISHED = {"refusal", "max_tokens", "pause_turn"}  # stop reasons of a response that is not worth keeping


def build_request(tier: str, request: dict) -> dict:
    """The tier's settings under the caller's, minus what the tier's model does not accept."""
    params = {**TIERS[tier], **request}
    if params["model"] in NO_EFFORT:
        config = {name: value for name, value in params.pop("output_config", {}).items() if name != "effort"}
        if config:
            params["output_config"] = config
    return params


def cost(response) -> dict:
    """Tokens and USD of a response, at the price of the model that answered (a fallback can change it)."""
    usage = response.usage
    # ponytail: prompt-cache writes and reads are charged at the plain input rate (writes cost more, reads far
    # less). Add per-model cache prices when a caller turns prompt caching on.
    tokens_in = usage.input_tokens + sum(getattr(usage, kind, 0) or 0
                                         for kind in ("cache_creation_input_tokens", "cache_read_input_tokens"))
    known = [name for name in USD_PER_MTOK if response.model.startswith(name)]
    # A model with no listed price is charged at the highest one, so a budget never under-counts.
    usd_in, usd_out = USD_PER_MTOK[max(known, key=len)] if known else max(USD_PER_MTOK.values())
    return {"model": response.model, "input_tokens": tokens_in, "output_tokens": usage.output_tokens,
            "usd": (tokens_in * usd_in + usage.output_tokens * usd_out) / 1e6}


def fingerprint(method: str, params: dict) -> str:
    """A hash of everything that decides the answer: the method, the model and the whole request."""
    def plain(value):  # output_format is a class; what the model sees of it is its JSON schema
        return TypeAdapter(value).json_schema() if isinstance(value, type) else repr(value)
    return hashlib.sha256(json.dumps([method, params], sort_keys=True, default=plain).encode()).hexdigest()


def restore(path: Path, output_format=None):
    """A stored response as the object the SDK returned the first time, with parsed_output rebuilt."""
    raw = path.read_text(encoding="utf-8")
    if output_format is None:
        return BetaMessage.model_validate_json(raw)
    message = ParsedBetaMessage[output_format].model_validate_json(raw)
    for block in message.content:
        if block.type == "text":
            block.parsed_output = TypeAdapter(output_format).validate_json(block.text)
    return message


class BudgetExceeded(RuntimeError):
    """The key has spent its budget. The model was not called."""


class Gateway:
    """Takes the arguments of client.beta.messages.create and .parse, plus the task, and returns the SDK's response."""

    def __init__(self, client, routes: dict[str, str | list[str]] | None = None, *, key: str = "default",
                 budget_usd: float | None = None, ledger: str | Path | None = None,
                 cache: str | Path | None = None, timeout: float | None = None):
        self.client = client
        # A route is the tier to use and, after it, the tiers to fall back to, in order.
        self.routes = {task: [route] if isinstance(route, str) else list(route)
                       for task, route in (routes or {}).items()}
        unknown = {tier for tiers in self.routes.values() for tier in tiers} - TIERS.keys()
        if unknown or [] in self.routes.values():
            raise ValueError(f"a route needs at least one of the tiers {sorted(TIERS)}; unknown: {sorted(unknown)}")
        self.timeout = timeout  # seconds per attempt; the SDK retries a timeout before the gateway falls back
        self.cache = Path(cache) if cache else None  # a folder with one JSON file per request
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.key = key  # who is spending: a run, a project, a user
        self.calls: list[dict] = []  # one record per call made through this gateway
        self.ledger = Path(ledger) if ledger else None
        self.budget_usd = budget_usd
        self.spent = 0.0  # USD spent by this key, counting earlier runs when there is a ledger
        if self.ledger:
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            if self.ledger.exists():
                lines = map(json.loads, self.ledger.read_text(encoding="utf-8").splitlines())
                self.spent = sum(line["usd"] for line in lines if line["key"] == key)

    def create(self, task: str, **request):
        return self._call("create", task, request)

    def parse(self, task: str, **request):
        return self._call("parse", task, request)

    def _call(self, method: str, task: str, request: dict):
        # ponytail: checked before the call, so a key can overshoot by one call, and by the calls in flight when
        # two processes share it. Reserve an estimate from count_tokens if a hard ceiling is ever needed.
        if self.budget_usd is not None and self.spent >= self.budget_usd:
            raise BudgetExceeded(f"key {self.key!r} has spent ${self.spent:.4f} of its ${self.budget_usd:.2f} budget")
        tiers = self.routes.get(task, ["large"])  # a task nobody has measured on the small model stays on the large one
        client = self.client.with_options(timeout=self.timeout) if self.timeout else self.client
        start = time.perf_counter()
        for position, tier in enumerate(tiers):
            params = build_request(tier, request)
            stored = self.cache / f"{fingerprint(method, params)}.json" if self.cache else None
            cached = stored is not None and stored.exists()
            try:
                response = (restore(stored, params.get("output_format")) if cached
                            else getattr(client.beta.messages, method)(**params))
            except (anthropic.RateLimitError, anthropic.APIConnectionError) as error:  # 429, no connection, timeout
                failure = error
            except anthropic.APIStatusError as error:
                if error.status_code < 500:  # a request the API rejects would be rejected on any model
                    raise
                failure = error
            else:
                if stored and not cached and response.stop_reason not in UNFINISHED:
                    stored.write_text(response.to_json(), encoding="utf-8")
                call = {"task": task, "tier": tier, **cost(response), "fallback": position > 0, "cached": cached,
                        "ms": round((time.perf_counter() - start) * 1000)}
                self._record({**call, "usd": 0.0} if cached else call)  # a hit keeps its tokens: what was saved
                return response
        raise failure

    def _record(self, call: dict) -> None:
        call = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "key": self.key, **call}
        self.calls.append(call)
        self.spent += call["usd"]
        if self.ledger:
            with self.ledger.open("a", encoding="utf-8") as file:
                file.write(json.dumps(call) + "\n")
