"""The console over the fake client: no test calls the API."""

from console import ANSI, ROUTES, WIDTH, Console, Fake
from model_gateway import Gateway


def console(**options) -> Console:
    return Console(Gateway(Fake(), ROUTES, **options), fake=True)


def test_each_agent_runs_on_the_tier_of_its_task_and_keeps_its_own_conversation():
    screen = console()
    screen.handle("@explorer what is a tier?")
    screen.handle("and a route?")  # plain text goes to the agent last addressed
    screen.handle("@main plan the release")
    assert [(call["task"], call["tier"]) for call in screen.gateway.calls] == [
        ("explore", "small"), ("explore", "small"), ("plan", "large")]
    assert len(screen.agents["explorer"]["messages"]) == 4
    assert len(screen.agents["main"]["messages"]) == 2


def test_a_spent_budget_is_reported_and_the_question_leaves_the_conversation():
    screen = console(budget_usd=0.0)
    assert screen.handle("@worker write a parser")
    assert screen.agents["worker"] == {"messages": [], "status": "failed"}
    assert "BudgetExceeded" in screen.log[-1]


def test_an_unknown_agent_calls_nothing_and_quit_leaves():
    screen = console()
    assert screen.handle("@nobody hello")
    assert not screen.gateway.calls and "nobody" in screen.log[-1]
    assert not screen.handle("/quit")


def test_no_line_of_the_tree_is_wider_than_the_screen(tmp_path):
    screen = console(budget_usd=0.25, cache=tmp_path / ("deep" * 20), ledger=tmp_path / ("deep" * 20) / "ledger.jsonl")
    screen.handle("@researcher " + "long text " * 40)
    assert max(len(ANSI.sub("", line)) for line in screen.render().splitlines()) <= WIDTH
