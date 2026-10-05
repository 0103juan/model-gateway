"""A terminal console for the gateway: the routing tree as calls go through it, and a prompt to talk to each agent.

    uv run console.py             # needs ANTHROPIC_API_KEY; spends from the key's budget
    uv run console.py --fake      # no key and no cost: canned answers through the real gateway

    @explorer what does the cache key cover?    talk to an agent; each one keeps its own conversation
    and why is the model part of it?            plain text goes to the agent you last talked to
    /clear   /quit

An agent is a name, a task and a system prompt. The gateway decides which model runs the task.
"""

import argparse
import re
import sys
import tempfile
import textwrap
import time
from types import SimpleNamespace

import anthropic
from anthropic.types.beta import BetaMessage

from model_gateway import TIERS, USD_PER_MTOK, BudgetExceeded, Gateway

SHORT = " Answer in plain text, in at most a few short paragraphs."
AGENTS = {  # name -> (task, what it is for, system prompt); the first one sits at the top of the tree
    "main": ("plan", "plans + answers", "You are the lead engineer of a small team. Plan the work and answer." + SHORT),
    "worker": ("write", "writes the code", "You write code. Give the code first, then one line on what it does."),
    "explorer": ("explore", "explains code", "You explain the code or the error the user pastes." + SHORT),
    "researcher": ("research", "sums up text", "You summarise the text the user pastes." + SHORT),
}
ROUTES = {"explore": ["small", "large"], "research": ["small", "large"], "write": ["large", "small"]}

WIDTH, SIDE = 90, 24
TREE = WIDTH - SIDE - 2
LARGE, SMALL, ROUTER, KEY, DIM = 117, 111, 114, 141, 244  # 256-colour codes
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def paint(text, colour) -> str:
    return f"\x1b[38;5;{colour}m{text}\x1b[0m"


def bold(text) -> str:
    return f"\x1b[1m{text}\x1b[0m"


def pad(text: str, width: int, centre=False) -> str:
    """To the given width on screen, whatever colour codes the text carries."""
    space = max(width - len(ANSI.sub("", text)), 0)
    left = space // 2 if centre else 0
    return " " * left + text + " " * (space - left)


def box(title: str, rows: list[str], width: int, colour: int, height: int = 0) -> list[str]:
    rows = rows + [""] * (height - 2 - len(rows))
    top = f"┌{f' {title} ' if title else '':─^{width - 2}}┐"
    side = paint("│", colour)
    return [paint(top, colour), *(side + pad(row, width - 2) + side for row in rows), paint(f"└{'─' * (width - 2)}┘", colour)]


def beside(blocks: list[list[str]], gap: int = 2) -> list[str]:
    widths = [len(ANSI.sub("", block[0])) for block in blocks]
    return [(" " * gap).join(pad(block[row] if row < len(block) else "", width) for block, width in zip(blocks, widths))
            for row in range(max(map(len, blocks)))]


def fan(centres: list[int], trunk: int, down: bool) -> list[str]:
    """The lines that join one box (at column trunk) to a row of boxes (at the given columns)."""
    line = [" "] * TREE
    for column in range(min(centres + [trunk]), max(centres + [trunk]) + 1):
        line[column] = "─"
    for column in centres:
        line[column] = "┬" if down else "┴"
    line[centres[0]], line[centres[-1]] = ("┌", "┐") if down else ("└", "┘")
    line[trunk] = "┼" if trunk in centres else "┴" if down else "┬"
    arrows = "".join("▼" if column in (centres if down else [trunk]) else " " for column in range(TREE))
    return [paint("".join(line), DIM), paint(arrows, DIM)]


def bar(share: float, width: int, colour: int) -> str:
    filled = round(min(max(share, 0), 1) * width)
    return paint("█" * filled, colour) + paint("░" * (width - filled), DIM)


def count(number: int, word: str) -> str:
    return f"{number} {word}{'' if number == 1 else 's'}"


class Fake:
    """Stands in for the SDK client, so the tree can be seen and tested with no key. It prices like the real one."""

    def __init__(self):
        self.beta = SimpleNamespace(messages=self)

    def with_options(self, **options):
        return self

    def create(self, **request):
        text = f"(fake answer, no model was called) You said: {request['messages'][-1]['content']}"
        return BetaMessage.model_validate({
            "id": "msg_fake", "type": "message", "role": "assistant", "model": request["model"],
            "stop_reason": "end_turn", "stop_sequence": None, "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": len(str(request)) // 4, "output_tokens": len(text) // 4}})


class Console:
    def __init__(self, gateway: Gateway, fake: bool = False):
        self.gateway = gateway
        self.fake = fake
        self.agents = {name: {"messages": [], "status": "idle"} for name in AGENTS}
        self.current = next(iter(AGENTS))
        self.log: list[str] = []

    def handle(self, line: str) -> bool:
        """One line typed at the prompt. False when it asks to leave."""
        line = line.strip()
        if line in ("/quit", "/exit"):
            return False
        if line == "/clear":
            self.agents[self.current] = {"messages": [], "status": "idle"}
        elif line.startswith("@"):
            name, _, text = line[1:].partition(" ")
            if name not in AGENTS:
                self.log.append(f"no agent {name!r}; there are: {', '.join(AGENTS)}")
            else:
                self.current = name
                if text.strip():
                    self.send(name, text.strip())
        elif line:
            self.send(self.current, line)
        return True

    def send(self, name: str, text: str) -> None:
        task, _, system = AGENTS[name]
        agent = self.agents[name]
        agent["messages"].append({"role": "user", "content": text})
        agent["status"] = "running"
        self.draw()
        try:
            # max_tokens caps what one turn can cost; the caller's settings win over the tier's
            response = self.gateway.create(task=task, system=system, messages=agent["messages"], max_tokens=2000)
            answer = "".join(block.text for block in response.content if block.type == "text")
            failure = "" if answer else f"no text in the answer (stop_reason {response.stop_reason})"
        except (BudgetExceeded, anthropic.APIError) as error:
            failure = f"{type(error).__name__}: {error}"
        if failure:  # the turn did not happen: the question leaves the conversation so the next one is well formed
            agent["messages"].pop()
            agent["status"] = "failed"
            self.log.append(f"{time.strftime('%H:%M:%S')}  {name:<11}{failure}")
            return
        agent["messages"].append({"role": "assistant", "content": answer})
        agent["status"] = "done"
        call = self.gateway.calls[-1]
        flags = " cached" * call["cached"] + " fallback" * call["fallback"]
        self.log.append(f"{call['ts'][11:19]}  {name:<11}{task} → {call['tier']} {call['model']}  "
                        f"{call['input_tokens']:,}+{call['output_tokens']:,} tok  ${call['usd']:.4f}  {call['ms']} ms{flags}")

    def draw(self) -> None:
        print("\x1b[2J\x1b[H" + self.render(), flush=True)

    def agent_box(self, name: str, width: int) -> list[str]:
        task, role, _ = AGENTS[name]
        tier = self.gateway.routes.get(task, ["large"])[0]
        model = TIERS[tier]["model"]
        usd_in, usd_out = USD_PER_MTOK[max((known for known in USD_PER_MTOK if model.startswith(known)), key=len)]
        agent = self.agents[name]
        mark = {"idle": "○ idle", "running": "◐ running", "failed": "✕ failed"}.get(
            agent["status"], "● " + count(len(agent["messages"]) // 2, "turn"))
        colour = LARGE if tier == "large" else SMALL
        rows = [bold(name) + (" ◂" if name == self.current else ""), paint(model, colour), role,
                paint(f"${usd_in:g} / ${usd_out:g} per 1M", DIM), paint(mark, colour if agent["status"] != "idle" else DIM)]
        return box("", [pad(row, width - 2, centre=True) for row in rows], width, colour)

    def render(self) -> str:
        gateway, calls = self.gateway, self.gateway.calls
        first, *others = AGENTS
        width = (TREE - 2 * (len(others) - 1)) // len(others)
        centres = [index * (width + 2) + width // 2 for index in range(len(others))]
        trunk = TREE // 2
        stem = [paint(" " * trunk + "│", DIM)]

        total = sum(call["usd"] for call in calls) or 1
        routes = []
        for task in dict.fromkeys(task for task, _, _ in AGENTS.values()):
            mine = [call for call in calls if call["task"] == task]
            chain = paint(" → ", DIM).join(paint(tier, LARGE if tier == "large" else SMALL)
                                           for tier in gateway.routes.get(task, ["large"]))
            usd = sum(call["usd"] for call in mine)
            routes.append(pad(f" {task}", 11) + pad(chain, 15) + bar(usd / total, 8, ROUTER) + f" ${usd:.4f}"
                          + f"{count(len(mine), 'call'):>9}" + f"{sum(call['cached'] for call in mine):>3} cached")
        routes.append(paint(f" no route → large · fallbacks on 429, 5xx, timeout: {sum(call['fallback'] for call in calls)}", DIM))

        stores = [f" ledger  {gateway.ledger.name[-24:] if gateway.ledger else 'off'}",
                  f" cache   {gateway.cache.name[-24:] if gateway.cache else 'off'} · {count(sum(call['cached'] for call in calls), 'hit')}"]
        tree = [pad(row, TREE, centre=True) for row in self.agent_box(first, 30)] + stem
        tree += box("GATEWAY · route by task", routes, TREE, ROUTER) + fan(centres, trunk, down=True)
        tree += beside([self.agent_box(name, width) for name in others]) + fan(centres, trunk, down=False)
        tree += [pad(row, TREE, centre=True) for row in box("every call is recorded", stores, 46, LARGE)]

        spent = f"${gateway.spent:.4f}"
        side = [f" spent {spent:>14}", paint(" all runs of this key", DIM), ""]
        if gateway.budget_usd is not None:
            share = gateway.spent / gateway.budget_usd if gateway.budget_usd else 1
            side = [f" budget {f'${gateway.budget_usd:.2f}':>13}", f" spent {spent:>14}",
                    " " + bar(share, 14, KEY) + f" {min(share, 9.99):>4.0%}", paint(" all runs of this key", DIM), "",
                    paint(" once spent, the", DIM), paint(" gateway stops before", DIM), paint(" calling the model", DIM), ""]
        side += [bold(" this session"), f" calls {len(calls):>14}", f" from cache {sum(call['cached'] for call in calls):>9}",
                 f" fallbacks {sum(call['fallback'] for call in calls):>10}",
                 f" tokens in {sum(call['input_tokens'] for call in calls):>10,}",
                 f" tokens out {sum(call['output_tokens'] for call in calls):>9,}", ""]
        if calls:
            last = calls[-1]
            side += [bold(" last call"), f" {last['task']} → {last['tier']}", f" {last['ms']} ms  ${last['usd']:.4f}"]

        title = bold("MODEL GATEWAY") + paint("  ·  ", DIM) + "  ".join(
            paint("■ ", colour) + label for colour, label in
            ((LARGE, f"large {TIERS['large']['model']}"), (SMALL, f"small {TIERS['small']['model']}"), (ROUTER, "router"), (KEY, "key")))
        lines = [title, paint("═" * WIDTH, DIM)]
        if self.fake:
            lines.append(paint("FAKE MODE · no model is called · the amounts are what these tokens would have cost", KEY))
        lines += beside([box(f"KEY · {gateway.key}"[:SIDE - 4], side, SIDE, KEY, height=len(tree)), tree])
        lines += box("session log", [" " + entry[:WIDTH - 4] for entry in self.log[-5:]] or [paint(" nothing yet", DIM)], WIDTH, DIM)

        for message in self.agents[self.current]["messages"][-2:]:
            who = "you" if message["role"] == "user" else self.current
            lines += ["", bold(f"{who} ›")] + [wrapped for paragraph in message["content"].splitlines()
                                              for wrapped in textwrap.wrap(paragraph, WIDTH) or [""]]
        lines += ["", paint(f"@{' @'.join(AGENTS)}  ·  /clear  ·  /quit", DIM)]
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fake", action="store_true", help="canned answers: no key, no cost; the cache is a temporary folder")
    parser.add_argument("--key", default="console", help="who is spending (default: console)")
    parser.add_argument("--budget", type=float, default=0.25, help="USD this key may spend across runs (default: 0.25)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.fake:
        gateway = Gateway(Fake(), ROUTES, key="fake", budget_usd=args.budget, cache=tempfile.mkdtemp())
    else:
        gateway = Gateway(anthropic.Anthropic(), ROUTES, key=args.key, budget_usd=args.budget,
                          ledger=".gateway/ledger.jsonl", cache=".gateway/cache", timeout=60)
    console = Console(gateway, fake=args.fake)
    try:
        while True:
            console.draw()
            if not console.handle(input(f"\n@{console.current} › ")):
                break
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
