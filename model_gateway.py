"""One entry point for model calls. The caller names the task; a table decides which model runs it,
and every call leaves a record of what it cost.

    gateway = Gateway(anthropic.Anthropic(), routes={"rewrite": "small"}, ledger="ledger.jsonl")
    response = gateway.parse(task="rewrite", system=..., messages=[...], output_format=Rewrite)
    gateway.calls[-1]["usd"]
"""

import json
import time
from pathlib import Path

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


class Gateway:
    """Takes the arguments of client.beta.messages.create and .parse, plus the task, and returns the SDK's response."""

    def __init__(self, client, routes: dict[str, str] | None = None, *, key: str = "default",
                 ledger: str | Path | None = None):
        self.client = client
        self.routes = routes or {}
        unknown = set(self.routes.values()) - TIERS.keys()
        if unknown:
            raise ValueError(f"unknown tiers in routes: {sorted(unknown)}; the tiers are {sorted(TIERS)}")
        self.key = key  # who is spending: a run, a project, a user
        self.calls: list[dict] = []  # one record per call made through this gateway
        self.ledger = Path(ledger) if ledger else None
        if self.ledger:
            self.ledger.parent.mkdir(parents=True, exist_ok=True)

    def create(self, task: str, **request):
        return self._call("create", task, request)

    def parse(self, task: str, **request):
        return self._call("parse", task, request)

    def _call(self, method: str, task: str, request: dict):
        tier = self.routes.get(task, "large")  # a task nobody has measured on the small model stays on the large one
        start = time.perf_counter()
        response = getattr(self.client.beta.messages, method)(**build_request(tier, request))
        self._record({"task": task, "tier": tier, **cost(response),
                      "ms": round((time.perf_counter() - start) * 1000)})
        return response

    def _record(self, call: dict) -> None:
        call = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "key": self.key, **call}
        self.calls.append(call)
        if self.ledger:
            with self.ledger.open("a", encoding="utf-8") as file:
                file.write(json.dumps(call) + "\n")
