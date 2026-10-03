"""One entry point for model calls. The caller names the task; a table decides which model runs it.

    gateway = Gateway(anthropic.Anthropic(), routes={"rewrite": "small"})
    response = gateway.parse(task="rewrite", system=..., messages=[...], output_format=Rewrite)
"""

# A tier is a model plus the request settings that model needs.
# A safety decline on the large model is re-run server-side on Anthropic's recommended fallback model.
TIERS = {
    "small": {"model": "claude-haiku-4-5", "max_tokens": 16000},
    "large": {"model": "claude-sonnet-5-5", "max_tokens": 16000,
              "betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"},
}
NO_EFFORT = {"claude-haiku-4-5"}  # models that reject output_config.effort


def build_request(tier: str, request: dict) -> dict:
    """The tier's settings under the caller's, minus what the tier's model does not accept."""
    params = {**TIERS[tier], **request}
    if params["model"] in NO_EFFORT:
        config = {name: value for name, value in params.pop("output_config", {}).items() if name != "effort"}
        if config:
            params["output_config"] = config
    return params


class Gateway:
    """Takes the arguments of client.beta.messages.create and .parse, plus the task, and returns the SDK's response."""

    def __init__(self, client, routes: dict[str, str] | None = None):
        self.client = client
        self.routes = routes or {}
        unknown = set(self.routes.values()) - TIERS.keys()
        if unknown:
            raise ValueError(f"unknown tiers in routes: {sorted(unknown)}; the tiers are {sorted(TIERS)}")

    def create(self, task: str, **request):
        return self._call("create", task, request)

    def parse(self, task: str, **request):
        return self._call("parse", task, request)

    def _call(self, method: str, task: str, request: dict):
        tier = self.routes.get(task, "large")  # a task nobody has measured on the small model stays on the large one
        return getattr(self.client.beta.messages, method)(**build_request(tier, request))
