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
