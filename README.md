# model-gateway

One entry point for the model calls of my AI projects. It picks a small or a large model for each task, falls
back when a model fails or stalls, caches responses so that re-running an evaluation costs almost nothing,
records what every call cost, and stops when a budget is spent.

**Status: design.** No code is merged yet. The plan is in [docs/design.md](docs/design.md), the choices behind it
in [docs/decisions.md](docs/decisions.md), and the work is tracked in the
[v0.1 milestone](https://github.com/0103juan/model-gateway/milestone/1).

## Why

My five AI repositories each call the Anthropic SDK on their own:

- The model settings and the price table are copied from repo to repo. The price table exists three times.
- Every stage of every pipeline uses the same model, including cheap stages such as rewriting a search query.
  Whether a smaller model does those stages as well has not been measured.
- An end-to-end evaluation costs between 0.25 and 0.66 USD per run
  ([enterprise-rag](https://github.com/0103juan/enterprise-rag),
  [consultor-tributario](https://github.com/0103juan/consultor-tributario)), and running it again pays again for
  requests that did not change.
- Nothing stops a run that spends more than intended.

## What v0.1 will do

```python
import anthropic
from model_gateway import Gateway

gateway = Gateway(
    anthropic.Anthropic(),
    routes={"rewrite": ["small", "large"], "generate": ["large"]},  # first choice, then fallbacks
    key="eval", budget_usd=1.00,                                    # stop when this key has spent a dollar
    ledger="ledger.jsonl", cache=".gateway/cache",
)
response = gateway.create(task="generate", system="...", messages=[...])
```

`response` is the SDK's own response object, so the calling code does not change.

v0.1 is done when two repositories use it and their READMEs report the cost and the answer quality of the small
model against the large one on the same evaluation set.

## Not in v0.1

Async clients, streaming, the SDK's tool runner, other providers, and running as a network proxy. The reasons are
in [docs/decisions.md](docs/decisions.md).

## Licence

MIT. See [LICENSE](LICENSE).
