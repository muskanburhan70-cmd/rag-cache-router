"""Cost-aware router: pick the cheapest model tier likely to answer well, within a budget."""
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    tier: str
    model: str
    in_per_mtok: float   # USD per million input tokens  (PLACEHOLDER - edit to match the pricing page)
    out_per_mtok: float  # USD per million output tokens (PLACEHOLDER)

    def cost(self, in_tokens: int, out_tokens: int) -> float:
        return (in_tokens * self.in_per_mtok + out_tokens * self.out_per_mtok) / 1_000_000


TIERS = {
    "small": ModelSpec("small", "claude-haiku-5-5", 1.0, 5.0),
    "medium": ModelSpec("medium", "claude-sonnet-5-5", 3.0, 15.0),
    "large": ModelSpec("large", "claude-opus-5-5", 5.0, 25.0),
}
ORDER = ["small", "medium", "large"]

REASONING = re.compile(r"\b(why|compare|versus|vs|trade-?off|difference|design|evaluate|analy[sz]e|pros and cons|step by step|derive|explain how)\b", re.I)
MULTIPART = re.compile(r"\b(and also|as well as|then|first|second|additionally)\b|[;?].+\?", re.I)


@dataclass
class RouteDecision:
    tier: str
    model: str
    score: float
    est_cost: float
    reasons: list[str] = field(default_factory=list)


class CostAwareRouter:
    def __init__(self, budget_usd: float | None = None, small_max: float = 0.35, medium_max: float = 0.65,
                 expected_out_tokens: int = 250):
        self.budget = budget_usd
        self.spent = 0.0
        self.small_max, self.medium_max = small_max, medium_max
        self.expected_out = expected_out_tokens

    def complexity(self, query: str, top_retrieval_score: float) -> tuple[float, list[str]]:
        score, why = 0.0, []
        words = len(query.split())
        if words > 25:
            score += 0.25; why.append("long query")
        elif words > 12:
            score += 0.10
        if REASONING.search(query):
            score += 0.35; why.append("needs reasoning/comparison")
        if MULTIPART.search(query):
            score += 0.20; why.append("multi-part question")
        if top_retrieval_score < 0.45:
            score += 0.20; why.append("weak retrieval match")
        return min(score, 1.0), why

    def choose(self, query: str, prompt_tokens: int, top_retrieval_score: float) -> RouteDecision:
        score, why = self.complexity(query, top_retrieval_score)
        tier = "small" if score <= self.small_max else "medium" if score <= self.medium_max else "large"

        # Budget guard: step down until the estimated cost fits what's left.
        if self.budget is not None:
            remaining = self.budget - self.spent
            while tier != "small" and TIERS[tier].cost(prompt_tokens, self.expected_out) > remaining:
                tier = ORDER[ORDER.index(tier) - 1]
                why.append(f"downgraded to {tier} (budget)")
        spec = TIERS[tier]
        return RouteDecision(tier, spec.model, score, spec.cost(prompt_tokens, self.expected_out), why)

    def record(self, cost: float):
        self.spent += cost
