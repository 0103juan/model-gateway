# model-gateway

One entry point for the model calls of my AI projects. It picks a small or a large model for each task, falls
back when a model fails or stalls, caches responses so that re-running an evaluation costs nothing, records what
every call cost, and stops when a budget is spent.

```python
import anthropic
from model_gateway import Gateway

gateway = Gateway(
    anthropic.Anthropic(),
    routes={"rewrite": ["small", "large"]},   # first choice, then fallbacks; other tasks use the large model
    key="eval", budget_usd=1.00,              # stop when this key has spent a dollar
    ledger="ledger.jsonl", cache=".gateway/cache", timeout=60,
)
response = gateway.create(task="generate", system="...", messages=[...])
gateway.calls[-1]   # {"task": "generate", "model": "claude-sonnet-5-5", "usd": ..., "cached": False, ...}
```

`create` and `parse` take the arguments of the SDK's `client.beta.messages.create` and `.parse`, plus `task`, and
return the SDK's own response object, so the calling code keeps reading `response.content` and
`response.parsed_output` as before.

## Why

My AI repositories each called the Anthropic SDK on their own. The model settings and the price table were
copied from repo to repo (the price table existed three times), every stage of every pipeline used the same
model whether or not it needed it, re-running an evaluation paid again for requests that had not changed, and
nothing stopped a run that spent more than intended.

## Measured

The first two consumers are [consultor-tributario](https://github.com/0103juan/consultor-tributario) and
[enterprise-rag](https://github.com/0103juan/enterprise-rag). On 3 October 2026 the evaluation of
`consultor-tributario` (33 questions; rewrite, generate and judge per question, plus a grader) went through the
gateway three times:

| | Large (Claude Sonnet 5.5) | Small (Claude Haiku 4.5) | Large, repeated |
|---|---|---|---|
| Answer correct | 28 of 33 | 27 of 33 | 28 of 33 |
| Groundedness | 100% | 96.0% | 100% |
| Evidence retrieved | 26 of 32 | 26 of 32 | 26 of 32 |
| Model calls, and how many came from the cache | 126, 0 | 137, 0 | 126, 126 |
| Cost | $0.66 | $0.27 | $0.00 |

- **Cost per call matched the code it replaced.** The ledger of the large run adds up to $0.6569; the same
  evaluation cost $0.66 the day before, counted by the consumer's own price table.
- **The repeated run made no API call** and returned the same results line by line.
- **The comparison cleared one stage for the small model, not three.** Rewriting retrieved the same evidence on
  either model at a third of the cost. Generation and judging on the small model exposed two defects in the
  consumer, now tracked there. The reading of the numbers, including why a one-question gap is noise on 33
  questions, is in the
  [consumer's README](https://github.com/0103juan/consultor-tributario#small-model-against-large-model).

## What it does

- **Routing by task.** `routes` maps a task to a tier. `small` is Claude Haiku 4.5 and `large` is Claude Sonnet
  5.5 with the server-side refusal fallback. A task with no route runs on the large model. A setting the chosen
  model does not take (`output_config.effort` on Haiku 4.5) is removed from the request.
- **Fallback.** A route can be a list. On a rate limit, a server error, a dropped connection or a timeout, after
  the SDK's own retries, the next tier takes the call, and the record says `fallback: true`. A request the API
  rejects is raised without trying another model. `timeout` sets the seconds each attempt may take.
- **Response cache.** With `cache=<folder>` a request that was already answered is read from disk: the record
  says `cached: true` and costs nothing. The key is a hash of the whole request, model included, so any change
  misses. A refusal or a truncated response is not stored. Delete the folder to start over.
- **Cost per call.** `gateway.calls` holds one record per call: key, task, tier, the model that answered,
  tokens, USD at list price and milliseconds. With `ledger=<file>` each record is also appended as a JSON line.
- **Budget per key.** With `budget_usd`, the gateway raises `BudgetExceeded` instead of calling the model once
  the key has spent its budget. The spend of a key is the sum of its ledger lines, so the limit holds across
  runs.
- **Async and agent loops.** `AsyncGateway` is the same class over `anthropic.AsyncAnthropic`. `tool_runner`
  takes the arguments of the SDK's `client.beta.messages.tool_runner`, plus `task`, and returns the SDK's
  runner: every turn of the loop is routed, cached, recorded and counted against the budget like any other call.

  ```python
  gateway = AsyncGateway(anthropic.AsyncAnthropic(), routes={"agent": ["large", "small"]}, ledger="ledger.jsonl")
  final = await gateway.tool_runner(task="agent", messages=[...], tools=[...]).until_done()
  sum(call["usd"] for call in gateway.calls)   # what the run cost
  ```

## Use it

```
uv add "model-gateway @ git+https://github.com/0103juan/model-gateway@v0.1.0"
```

To work on it:

```
uv sync
uv run pytest     # 32 tests, no API key; the cache and tool-runner tests run the real SDK over a fake HTTP transport
```

## Limits

- **Fallback and the budget limit have not happened against the API.** In the three runs no call failed and no
  run reached its limit. Both are covered by tests that raise the SDK's own exceptions.
- The budget is checked before each call, so a key can overshoot by one call, and by the calls in flight if two
  processes share a key.
- Prompt-cache writes and reads are priced at the plain input rate. Neither consumer uses prompt caching.
- When the server-side refusal fallback answers, the whole call is priced at the model that answered.
- While a cache file exists, the same request always returns the same answer. A caller that wants several
  samples of one prompt has to leave the cache off.
- That Haiku 4.5 rejects `output_config.effort` comes from the API documentation. The gateway removes the
  setting and those requests were accepted; sending it was not tried.
- No streaming. `AsyncGateway` and `tool_runner` exist since this version, but the two MCP chat clients and the
  agent do not use them yet ([#10](https://github.com/0103juan/model-gateway/issues/10)), and neither has run
  against the API: the tests drive the real SDK runner over a fake HTTP transport.
- `tool_runner` puts the gateway inside the SDK's runner by setting a private attribute of it, because the
  runner offers no public way to change where it sends a turn. A release of the SDK can break that; the tests
  would fail.
- One provider. The prices are a table in the code, dated, and have to be updated by hand.

The design is in [docs/design.md](docs/design.md) and the choices behind it in
[docs/decisions.md](docs/decisions.md).

## Licence

MIT. See [LICENSE](LICENSE).
