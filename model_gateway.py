"""One entry point for model calls. The caller names the task; a table decides which model runs it and which
one takes over if it fails, a request already answered is served from disk, every call leaves a record of
what it cost, and a key that has spent its budget gets no more calls.

    gateway = Gateway(anthropic.Anthropic(), routes={"rewrite": ["small", "large"]}, key="eval", budget_usd=1.00,
                      ledger="ledger.jsonl", cache=".gateway/cache", timeout=60)
    response = gateway.parse(task="rewrite", system=..., messages=[...], output_format=Rewrite)
    gateway.calls[-1]["usd"]

AsyncGateway is the same over anthropic.AsyncAnthropic, and tool_runner sends every turn of an agent loop
through the same steps. GeminiClient puts Gemini on Vertex AI behind the same calls:

    gateway = Gateway(GeminiClient(project="my-project"), tiers=GEMINI_TIERS, routes=..., budget_usd=...)
"""

import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import anthropic
from anthropic.types.beta import BetaMessage
from anthropic.types.beta.parsed_beta_message import ParsedBetaMessage
from pydantic import TypeAdapter, ValidationError

# A tier is a model plus the request settings that model needs.
# A safety decline on the large model is re-run server-side on Anthropic's recommended fallback model.
TIERS = {
    "small": {"model": "claude-haiku-4-5", "max_tokens": 16000},
    "large": {"model": "claude-sonnet-5-5", "max_tokens": 16000,
              "betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"},
}
NO_EFFORT = {"claude-haiku-4-5"}  # models that reject output_config.effort
GEMINI_TIERS = {
    "small": {"model": "gemini-3.5-flash-lite", "max_tokens": 16000},
    "large": {"model": "gemini-3.5-flash", "max_tokens": 16000},  # Gemini's thinking counts against max_tokens
}

# USD per million tokens (input, output), list price on 25 September 2026 (Gemini: the Vertex AI catalogue,
# global endpoint, 20 October 2026). A name also covers its dated snapshots and point releases
# ("claude-haiku-4-5-20251001", "claude-fable-5-1"); the longest match wins, so a Lite model is not priced as Flash.
USD_PER_MTOK = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.5-flash": (1.50, 9.00),
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


def build_request(tier: str, request: dict, tiers: dict = TIERS) -> dict:
    """The tier's settings under the caller's, minus what the tier's model does not accept."""
    params = {**tiers[tier], **request}
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


class Unavailable(RuntimeError):
    """A backend other than Anthropic could not answer now (rate limit, server error, timeout): another tier may."""


EFFORT_TO_THINKING = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH", "xhigh": "HIGH", "max": "HIGH"}
FINISH_TO_STOP = {"STOP": "end_turn", "MAX_TOKENS": "max_tokens"}  # any other reason (safety, recitation...) declines
# Retried, then a reason to try another tier: timeout, rate limit, cancelled by the server; and every 5xx.
RETRIED = (408, 429, 499, 500, 502, 503, 504)


class GeminiClient:
    """Gemini on Vertex AI behind the part of client.beta.messages the gateway calls: create and parse, with text
    in and text or a Pydantic model out. It returns the SDK's own message types, so routing, fallback, the cache,
    cost and the budget work unchanged. Anthropic's output_config.effort becomes Gemini's thinking level, and
    the thinking tokens are counted as output, which is how they are billed.

    ponytail: no tools, no streaming and no async client. Translate tool definitions and add a client.aio twin
    when an agent has to run on Gemini through the gateway."""

    def __init__(self, project: str | None = None, location: str = "global", *, client=None, timeout: float | None = None,
                 attempts: int = 3):
        if client is None:
            from google import genai  # pip install "model-gateway[gemini]"
            client = genai.Client(vertexai=True, project=project, location=location)
        # attempts: tries of one call before the gateway moves on, waiting up to a minute between them. Raise
        # it for long batch runs, where a per-minute quota is the usual 429.
        self._genai, self._timeout, self._attempts = client, timeout, attempts
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create, parse=self._parse))

    def with_options(self, timeout: float | None = None) -> "GeminiClient":
        return GeminiClient(client=self._genai, timeout=timeout, attempts=self._attempts)

    def _create(self, **request):
        return self._send(request, None)

    def _parse(self, output_format=None, **request):
        return self._send(request, output_format)

    def _send(self, request: dict, output_format):
        import httpx
        from google.genai import errors, types

        if {"tools", "tool_choice", "stream"} & request.keys():
            raise ValueError("GeminiClient sends text and structured output only: no tools, no streaming")

        def texts(content) -> list[str]:
            if isinstance(content, str):
                return [content]
            if any(block.get("type") != "text" for block in content):
                raise ValueError("GeminiClient sends text blocks only")
            return [block["text"] for block in content]

        system = request.get("system")
        effort = (request.get("output_config") or {}).get("effort")
        config = types.GenerateContentConfig(
            system_instruction="\n\n".join(texts(system)) if system else None,
            max_output_tokens=request.get("max_tokens"),
            temperature=request.get("temperature"),
            thinking_config=types.ThinkingConfig(thinking_level=EFFORT_TO_THINKING[effort]) if effort else None,
            response_mime_type="application/json" if output_format else None,
            response_schema=output_format,
            # Retried here before the gateway moves to another tier, as the Anthropic SDK does (three attempts).
            http_options=types.HttpOptions(timeout=int(self._timeout * 1000) if self._timeout else None,
                                           retry_options=types.HttpRetryOptions(attempts=self._attempts, max_delay=60,
                                                                                http_status_codes=list(RETRIED))))
        contents = [types.Content(role="model" if message["role"] == "assistant" else "user",
                                  parts=[types.Part(text=text) for text in texts(message["content"])])
                    for message in request["messages"]]
        try:
            response = self._genai.models.generate_content(model=request["model"], contents=contents, config=config)
        except errors.APIError as error:
            if error.code in RETRIED or error.code >= 500:
                raise Unavailable(f"{request['model']}: {error.code} {error.message}") from error
            raise
        except httpx.TransportError as error:  # no connection, timeout
            raise Unavailable(f"{request['model']}: {error!r}") from error

        usage = response.usage_metadata
        finish = response.candidates[0].finish_reason if response.candidates else None
        text = response.text or ""
        message = {"id": response.response_id or "", "type": "message", "role": "assistant",
                   "model": response.model_version or request["model"],
                   "content": [{"type": "text", "text": text}],
                   "stop_reason": FINISH_TO_STOP.get(finish.name if finish else "", "refusal"), "stop_sequence": None,
                   "usage": {"input_tokens": usage.prompt_token_count or 0,
                             "output_tokens": (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)}}
        if output_format is None:
            return BetaMessage.model_validate(message)
        parsed = ParsedBetaMessage[output_format].model_validate(message)
        try:
            parsed.content[0].parsed_output = TypeAdapter(output_format).validate_json(text)
        except ValidationError:  # a cut or declined answer: parsed_output stays None, as with Claude
            pass
        return parsed


class Gateway:
    """Takes the arguments of client.beta.messages.create and .parse, plus the task, and returns the SDK's response."""

    def __init__(self, client, routes: dict[str, str | list[str]] | None = None, *, key: str = "default",
                 budget_usd: float | None = None, ledger: str | Path | None = None,
                 cache: str | Path | None = None, timeout: float | None = None, tiers: dict = TIERS):
        self.client = client
        self.tiers = tiers  # which model each tier is: TIERS for Claude, GEMINI_TIERS for GeminiClient
        # A route is the tier to use and, after it, the tiers to fall back to, in order.
        self.routes = {task: [route] if isinstance(route, str) else list(route)
                       for task, route in (routes or {}).items()}
        unknown = {tier for tiers in self.routes.values() for tier in tiers} - self.tiers.keys()
        if unknown or [] in self.routes.values():
            raise ValueError(f"a route needs at least one of the tiers {sorted(self.tiers)}; unknown: {sorted(unknown)}")
        self.timeout = timeout  # seconds per attempt; the SDK retries a timeout before the gateway falls back
        self.cache = Path(cache) if cache else None  # a folder with one JSON file per request
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.key = key  # who is spending: a run, a project, a user
        self.calls: list[dict] = []  # one record per call made through this gateway
        self._lock = threading.Lock()  # threads sharing a gateway record their calls one at a time
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

    def tool_runner(self, task: str, **request):
        """client.beta.messages.tool_runner, with every turn of the loop sent through the gateway."""
        if request.get("stream"):
            raise ValueError("the gateway does not stream")
        # The runner wants a model up front, and it sends betas as a header that it would keep on every tier.
        tier = {name: value for name, value in self.tiers[self._tiers(task)[0]].items() if name != "betas"}
        runner = self.client.beta.messages.tool_runner(**{**tier, **request})
        owned = tier.keys() - request.keys()  # the first tier's settings: a turn takes them from the tier that runs it

        def turn(**params):
            return self._call("parse", task, {name: value for name, value in params.items() if name not in owned
                                              and not isinstance(value, (anthropic.Omit, anthropic.NotGiven))})

        # ponytail: the runner has no public way to take another client, so this sets a private attribute of
        # the SDK. The tool-runner tests drive the real runner and fail if a release renames it.
        runner._client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(parse=turn)))
        return runner

    def _tiers(self, task: str) -> list[str]:
        return self.routes.get(task, ["large"])  # a task nobody has measured on the small model stays on the large one

    def _send(self, method: str):
        client = self.client.with_options(timeout=self.timeout) if self.timeout else self.client
        return getattr(client.beta.messages, method)

    def _call(self, method: str, task: str, request: dict):
        plan, send = self._plan(method, task, request), self._send(method)
        try:
            params = next(plan)
            while True:
                try:
                    response = send(**params)
                except Exception as error:
                    params = plan.throw(error)
                else:
                    params = plan.send(response)
        except StopIteration as done:
            return done.value

    def _plan(self, method: str, task: str, request: dict):
        """One call as a generator, so the sync and the async gateway share every step but the sending: it
        yields each request to send and is sent the response, or thrown the error. It returns the response."""
        # ponytail: checked before the call, so a key can overshoot by one call, and by the calls in flight when
        # two processes share it. Reserve an estimate from count_tokens if a hard ceiling is ever needed.
        if self.budget_usd is not None and self.spent >= self.budget_usd:
            raise BudgetExceeded(f"key {self.key!r} has spent ${self.spent:.4f} of its ${self.budget_usd:.2f} budget")
        start = time.perf_counter()
        for position, tier in enumerate(self._tiers(task)):
            params = build_request(tier, request, self.tiers)
            stored = self.cache / f"{fingerprint(method, params)}.json" if self.cache else None
            cached = stored is not None and stored.exists()
            try:
                response = restore(stored, params.get("output_format")) if cached else (yield params)
            except (anthropic.RateLimitError, anthropic.APIConnectionError, Unavailable) as error:  # 429, no connection, timeout
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
        with self._lock:  # without it, concurrent appends on Windows interleave and leave broken lines
            self.calls.append(call)
            self.spent += call["usd"]
            if self.ledger:
                with self.ledger.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(call) + "\n")


class AsyncGateway(Gateway):
    """The same gateway over anthropic.AsyncAnthropic: create and parse are awaited, and tool_runner is
    iterated with `async for`."""

    async def _call(self, method: str, task: str, request: dict):
        plan, send = self._plan(method, task, request), self._send(method)
        try:
            params = next(plan)
            while True:
                try:
                    response = await send(**params)
                except Exception as error:
                    params = plan.throw(error)
                else:
                    params = plan.send(response)
        except StopIteration as done:
            return done.value
