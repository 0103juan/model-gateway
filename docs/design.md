# Design

Written on 3 October 2026, before any code.

## The problem

Five repositories call the Anthropic SDK directly. Each one carries its own copy of the model settings and of the
price table, uses one model for every stage, pays again for identical requests when an evaluation is re-run, and
has no spending limit. The two RAG pipelines make three kinds of call per question (rewrite the query, write the
answer, audit the answer), and an evaluation adds a fourth (grade the answer).

## The shape

A Python library with one class. The caller names the task; the gateway decides the rest.

```python
gateway = Gateway(client, routes={"rewrite": ["small", "large"]}, key="eval", budget_usd=1.00,
                  ledger="ledger.jsonl", cache=".gateway/cache", timeout=60)
response = gateway.parse(task="rewrite", system=..., messages=[...], output_format=Rewrite)
```

`create` and `parse` take the same arguments as `client.beta.messages.create` and `.parse`, plus `task`, and
return the SDK's response object. A call goes through these steps:

1. **Budget.** If the key has already spent its budget, raise `BudgetExceeded` without calling the model.
2. **Route.** Look up the task in `routes` to get an ordered list of tiers. A tier (`small`, `large`) is a model
   plus the request settings that model needs. A setting the model rejects is removed: the small model does
   not accept `output_config.effort`.
3. **Cache.** Hash the full request. If a response for that hash is on disk, return it at no cost.
4. **Call, then fall back.** Call the first tier. On a rate limit, a server error or a timeout, after the SDK's
   own retries, call the next tier. A request the API rejects (a 4xx) is raised: it would fail on any model.
5. **Record.** Append one line to the ledger: key, task, tier, the model that answered, tokens, USD,
   milliseconds, whether it came from the cache and whether it was a fallback.

## What it stores

- **Ledger:** one JSON object per line. The budget of a key is the sum of its lines, so it holds across runs.
- **Cache:** one JSON file per request hash, holding the response as the SDK serialises it.

Both are plain files that can be read, diffed and deleted by hand.

## How it will be checked

- Unit tests with a fake client: no test calls the API.
- CI on every pull request.
- The measurement that closes v0.1: the evaluation of `consultor-tributario` run twice through the gateway,
  once with every stage on the small model and once on the large one, reporting correct answers, groundedness
  and USD for each. A third run of an unchanged evaluation has to cost close to zero.

## Out of scope for v0.1

- Async clients and the SDK's tool runner (the two MCP chat clients and the agent use them). Planned for v0.2.
- Streaming.
- Other providers.
- A network proxy with its own keys and users.
