"""LLM backends. Uses the Claude API if ANTHROPIC_API_KEY is set, otherwise a free offline mock."""
import os
import re
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    in_tokens: int
    out_tokens: int


def est_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


class AnthropicLLM:
    name = "anthropic"

    def __init__(self):
        import anthropic

        self.client = anthropic.Anthropic()

    def generate(self, model: str, system: str, prompt: str, max_tokens: int = 400) -> LLMResponse:
        r = self.client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return LLMResponse(r.content[0].text, r.usage.input_tokens, r.usage.output_tokens)


class MockLLM:
    """Extractive stand-in so the whole pipeline (cache, routing, cost tracking) runs with no API key."""

    name = "mock"

    def generate(self, model: str, system: str, prompt: str, max_tokens: int = 400) -> LLMResponse:
        ctx = prompt.split("Context:")[-1].split("Question:")[0].strip()
        first = re.split(r"(?<=[.!?])\s+", ctx)[:2]
        text = f"[mock:{model}] " + " ".join(first)
        return LLMResponse(text, est_tokens(system + prompt), est_tokens(text))


def get_llm():
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLM()
        except Exception as e:
            print(f"[llm] anthropic unavailable ({type(e).__name__}); using mock")
    return MockLLM()
