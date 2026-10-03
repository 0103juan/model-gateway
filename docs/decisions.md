# Decisions

One short note per choice, newest last.

## 2026-10-03 · A library, not a proxy server

The callers are five Python scripts run by one person. A proxy would add a process to start, a port, and a
second set of keys, to solve problems (many teams, many languages) this project does not have. A proxy can be
built later on top of the same class.

## 2026-10-03 · Wrap the official SDK and return its objects

`create` and `parse` pass their arguments to the SDK and return what it returns. The callers already read
`response.content`, `response.parsed_output` and `response.usage`, so adopting the gateway changes one line per
call. A provider-neutral response type would mean rewriting every caller for a second provider nobody uses.

## 2026-10-03 · Routes are a table, not a classifier

The caller names the task and a dictionary maps it to a model. The pipelines already know which stage they are
in, so guessing the difficulty of a request from its text would add a model call and a source of error to
answer a question the code can answer for free.

## 2026-10-03 · Two tiers: Claude Haiku 4.5 and Claude Sonnet 5.5

The repositories already use Sonnet 5.5 for everything. Haiku 4.5 costs half as much per token. Whether it is
good enough for each stage is what the v0.1 measurement has to show; until then no route defaults to it.

## 2026-10-03 · Files, not a database

The ledger is a JSON Lines file and the cache is a folder of JSON files. One process writes them at a time, and
being able to open them in an editor is worth more here than concurrent writers. The limit is known: two
processes sharing a key can each overspend by the calls in flight.

## 2026-10-03 · Install from a git tag

`pip install git+https://github.com/0103juan/model-gateway@v0.1.0`. The package has one author and five
consumers in the same account, so a PyPI name would be publishing for its own sake.

## 2026-10-03 · A setting the model rejects is removed, not raised

Haiku 4.5 rejects a request that carries `output_config.effort`, according to the API documentation; this has
not been tried against the API yet. The callers set an effort per stage and should not have to know which model
a route points to, so the gateway drops the setting for that model. The list of such settings lives next to the
tiers, in one place.

## 2026-10-03 · Fall back only on a failure another model can fix

A rate limit, a server error or a timeout says nothing about the request, so the next tier may well answer it. A
400 or a 404 is a mistake in the request or in the tier table: sending it to a second model would hide the
mistake. The gateway does not retry on its own. The SDK already retries twice with backoff, and the fallback
starts after that, so with `timeout=60` the worst case before the next tier is three minutes.

A fallback from the large model to the small one changes the quality of the answer without telling the caller.
That is why no route falls back unless it lists a second tier, and why the record of every call says which
model answered and whether it was a fallback.

## 2026-10-03 · The cache key is the whole request, and entries do not expire

The cache exists so that re-running an evaluation that did not change is free. Any difference in the request
(model, prompt, schema, a setting) is a different key, so a stale answer can only come from the same question
asked the same way. Nothing expires: the entries are files, and deleting the folder is the invalidation.

The cost is that the same request always gets the same answer. A caller that wants several samples of one
prompt has to leave the cache off for that call path. A refusal or a truncated response is not stored, because
the next attempt may do better.
