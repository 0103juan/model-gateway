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

## See it

`console.py` draws the gateway as a tree and lets you talk to four agents from one prompt. An agent is a name,
a task and a system prompt; the gateway picks the model for the task. Each agent keeps its own conversation:
`@explorer <message>` talks to one, plain text goes to the one you last addressed.

```
uv run console.py --fake      # no key, no cost: canned answers through the real gateway
uv run console.py             # needs ANTHROPIC_API_KEY; the key "console" may spend $0.25 across runs
```

The screen after four messages in fake mode (the second one repeats the first, so it comes from the cache):

```
MODEL GATEWAY  ·  ■ large claude-sonnet-5-5  ■ small claude-haiku-4-5  ■ router  ■ key
══════════════════════════════════════════════════════════════════════════════════════════
FAKE MODE · no model is called · the amounts are what these tokens would have cost
┌───── KEY · fake ─────┐                   ┌────────────────────────────┐
│ budget         $0.25 │                   │            main            │
│ spent        $0.0007 │                   │     claude-sonnet-5-5      │
│ ░░░░░░░░░░░░░░   0%  │                   │      plans + answers       │
│ all runs of this key │                   │      $2 / $10 per 1M       │
│                      │                   │          ● 1 turn          │
│ once spent, the      │                   └────────────────────────────┘
│ gateway stops before │                                  │
│ calling the model    │  ┌────────────────── GATEWAY · route by task ───────────────────┐
│                      │  │ plan      large          ███░░░░░ $0.0003   1 call  0 cached │
│ this session         │  │ write     large → small  ███░░░░░ $0.0003   1 call  0 cached │
│ calls              4 │  │ explore   small → large  ██░░░░░░ $0.0002  2 calls  1 cached │
│ from cache         1 │  │ research  small → large  ░░░░░░░░ $0.0000  0 calls  0 cached │
│ fallbacks          0 │  │ no route → large · fallbacks on 429, 5xx, timeout: 0         │
│ tokens in        268 │  └──────────────────────────────────────────────────────────────┘
│ tokens out        66 │            ┌─────────────────────┼─────────────────────┐
│                      │            ▼                     ▼                     ▼
│ last call            │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ write → large        │  │     worker ◂     │  │     explorer     │  │    researcher    │
│ 1 ms  $0.0003        │  │claude-sonnet-5-5 │  │ claude-haiku-4-5 │  │ claude-haiku-4-5 │
│                      │  │ writes the code  │  │  explains code   │  │   sums up text   │
│                      │  │ $2 / $10 per 1M  │  │  $1 / $5 per 1M  │  │  $1 / $5 per 1M  │
│                      │  │     ● 1 turn     │  │     ● 1 turn     │  │      ○ idle      │
│                      │  └──────────────────┘  └──────────────────┘  └──────────────────┘
│                      │            └─────────────────────┼─────────────────────┘
│                      │                                  ▼
│                      │           ┌────────── every call is recorded ──────────┐
│                      │           │ ledger  off                                │
│                      │           │ cache   tmphwhicfjk · 1 hit                │
└──────────────────────┘           └────────────────────────────────────────────┘
┌───────────────────────────────────── session log ──────────────────────────────────────┐
│ 02:10:00  explorer   explore → small claude-haiku-4-5  61+18 tok  $0.0002  39 ms       │
│ 02:10:00  explorer   explore → small claude-haiku-4-5  61+18 tok  $0.0000  1 ms cached │
│ 02:10:00  main       plan → large claude-sonnet-5-5  80+15 tok  $0.0003  1 ms          │
│ 02:10:00  worker     write → large claude-sonnet-5-5  66+15 tok  $0.0003  1 ms         │
└────────────────────────────────────────────────────────────────────────────────────────┘

you ›
write a retry loop

worker ›
(fake answer, no model was called) You said: write a retry loop

@main @worker @explorer @researcher  ·  /clear  ·  /quit
```

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

## Use it

```
uv add "model-gateway @ git+https://github.com/0103juan/model-gateway@v0.1.0"
```

To work on it:

```
uv sync
uv run pytest     # 32 tests, no API key; the cache tests run the real SDK over a fake HTTP transport
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
- **The console has only run in fake mode.** Its agents are conversations: they have no tools, do not call each
  other and do not stream; the screen is redrawn after each answer.
- Sync `create` and `parse` only. Async clients, streaming and the SDK's tool runner are not covered, so the two
  MCP chat clients and the agent do not use the gateway yet
  ([#10](https://github.com/0103juan/model-gateway/issues/10)).
- One provider. The prices are a table in the code, dated, and have to be updated by hand.

The design is in [docs/design.md](docs/design.md) and the choices behind it in
[docs/decisions.md](docs/decisions.md).

## Licence

MIT. See [LICENSE](LICENSE).
