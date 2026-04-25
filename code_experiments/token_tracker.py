"""
token_tracker.py
----------------
A utility for measuring and reporting Claude API token usage and costs.
Designed for use in LLM efficiency experiments.

Usage:
    tracker = TokenTracker(model="claude-sonnet-4-6")
    
    with tracker.measure("my_experiment"):
        response = tracker.client.messages.create(...)
    
    tracker.report()
"""

import os
import time
import json
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from typing import Optional
import anthropic
from dotenv import load_dotenv



load_dotenv()


# -------------------------------------------------------------------
# Pricing (USD per million tokens) — verified April 2026
# Update here if Anthropic changes rates
# -------------------------------------------------------------------
PRICING = {
    "claude-opus-4-7":    {"input": 5.00,  "output": 25.00, "cache_write": 6.25,  "cache_read": 0.50},
    "claude-opus-4-6":    {"input": 5.00,  "output": 25.00, "cache_write": 6.25,  "cache_read": 0.50},
    "claude-sonnet-4-6":  {"input": 3.00,  "output": 15.00, "cache_write": 3.75,  "cache_read": 0.30},
    "claude-haiku-4-5":   {"input": 1.00,  "output": 5.00,  "cache_write": 1.25,  "cache_read": 0.10},
}


@dataclass
class Measurement:
    label: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    latency_ms: float = 0.0
    tool_calls: list = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        rates = PRICING.get(self.model, PRICING["claude-sonnet-4-6"])
        cost = (
            (self.input_tokens / 1_000_000) * rates["input"]
            + (self.output_tokens / 1_000_000) * rates["output"]
            + (self.cache_creation_tokens / 1_000_000) * rates["cache_write"]
            + (self.cache_read_tokens / 1_000_000) * rates["cache_read"]
        )
        return cost

    @property
    def cost_per_1k_calls_usd(self) -> float:
        """Useful for estimating production costs at scale."""
        return self.cost_usd * 1000


class TokenTracker:
    """
    Wraps the Anthropic client to track token usage and costs across
    multiple labelled experiments in a single session.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: Optional[str] = None):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self.measurements: list[Measurement] = []
        self._current: Optional[Measurement] = None

    @contextmanager
    def measure(self, label: str):
        """
        Context manager that records token usage for everything inside the block.
        
        Example:
            with tracker.measure("date_parsing_via_llm"):
                response = tracker.client.messages.create(...)
                tracker.record(response)
        """
        self._current = Measurement(label=label, model=self.model)
        self._start = time.perf_counter()
        try:
            yield self._current
        finally:
            self._current.latency_ms = (time.perf_counter() - self._start) * 1000
            self.measurements.append(self._current)
            self._current = None

    def record(self, response: anthropic.types.Message) -> anthropic.types.Message:
        """
        Record token usage from a response object.
        Call this inside a measure() block.
        Returns the response unchanged so you can chain it.
        
        Example:
            with tracker.measure("my_task"):
                response = tracker.record(
                    tracker.client.messages.create(...)
                )
                text = response.content[0].text
        """
        if self._current is None:
            raise RuntimeError("record() must be called inside a measure() block")

        usage = response.usage
        self._current.input_tokens += usage.input_tokens
        self._current.output_tokens += usage.output_tokens
        self._current.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self._current.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

        # Capture any tool calls made
        for block in response.content:
            if block.type == "tool_use":
                self._current.tool_calls.append({
                    "tool": block.name,
                    "input_keys": list(block.input.keys()) if isinstance(block.input, dict) else []
                })

        return response

    def record_turn(self, response: anthropic.types.Message) -> anthropic.types.Message:
        """Alias for record() — more readable in multi-turn loops."""
        return self.record(response)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, show_scale: bool = True):
        """Print a formatted summary table of all measurements."""
        if not self.measurements:
            print("No measurements recorded.")
            return

        col_w = 32
        print("\n" + "═" * 100)
        print(f"{'TOKEN USAGE REPORT':^100}")
        print(f"{'Model: ' + self.model:^100}")
        print("═" * 100)
        print(
            f"{'Experiment':<{col_w}} {'Input':>8} {'Output':>8} {'Cache W':>8} "
            f"{'Cache R':>8} {'Total':>8} {'Latency':>10} {'Cost (USD)':>12}"
        )
        print("─" * 100)

        total_input = total_output = total_cache_w = total_cache_r = total_cost = 0.0

        for m in self.measurements:
            tool_str = f"  ← tools: {', '.join(t['tool'] for t in m.tool_calls)}" if m.tool_calls else ""
            print(
                f"{m.label:<{col_w}} {m.input_tokens:>8,} {m.output_tokens:>8,} "
                f"{m.cache_creation_tokens:>8,} {m.cache_read_tokens:>8,} "
                f"{m.total_tokens:>8,} {m.latency_ms:>9.0f}ms "
                f"${m.cost_usd:>11.6f}{tool_str}"
            )
            total_input += m.input_tokens
            total_output += m.output_tokens
            total_cache_w += m.cache_creation_tokens
            total_cache_r += m.cache_read_tokens
            total_cost += m.cost_usd

        print("─" * 100)
        total_tokens = int(total_input + total_output)
        print(
            f"{'TOTAL':<{col_w}} {int(total_input):>8,} {int(total_output):>8,} "
            f"{int(total_cache_w):>8,} {int(total_cache_r):>8,} "
            f"{total_tokens:>8,} {'':>10} ${total_cost:>11.6f}"
        )

        if show_scale:
            print("\n  Scale projections:")
            print(f"    1,000 calls like this session → ${total_cost * 1000:,.2f}")
            print(f"   10,000 calls like this session → ${total_cost * 10000:,.2f}")

        print("═" * 100 + "\n")

    def compare(self, label_a: str, label_b: str):
        """Print a direct cost and token comparison between two labelled experiments."""
        a = next((m for m in self.measurements if m.label == label_a), None)
        b = next((m for m in self.measurements if m.label == label_b), None)

        if not a or not b:
            missing = label_a if not a else label_b
            print(f"Measurement '{missing}' not found.")
            return

        print(f"\n{'─'*60}")
        print(f"  Comparing: '{label_a}'  vs  '{label_b}'")
        print(f"{'─'*60}")
        print(f"  {'':30} {label_a:>12}  {label_b:>12}")
        print(f"  {'Input tokens':30} {a.input_tokens:>12,}  {b.input_tokens:>12,}")
        print(f"  {'Output tokens':30} {a.output_tokens:>12,}  {b.output_tokens:>12,}")
        print(f"  {'Total tokens':30} {a.total_tokens:>12,}  {b.total_tokens:>12,}")
        print(f"  {'Cost (USD)':30} ${a.cost_usd:>11.6f}  ${b.cost_usd:>11.6f}")
        print(f"  {'Latency':30} {a.latency_ms:>10.0f}ms  {b.latency_ms:>10.0f}ms")

        if a.cost_usd > 0 and b.cost_usd > 0:
            ratio = a.cost_usd / b.cost_usd
            cheaper = label_b if ratio > 1 else label_a
            print(f"\n  → '{cheaper}' is {max(ratio, 1/ratio):.1f}x cheaper")
            savings_pct = abs(1 - (b.cost_usd / a.cost_usd)) * 100
            print(f"  → Savings: {savings_pct:.1f}% per call")
            print(f"  → Savings at 10,000 calls: ${abs(a.cost_usd - b.cost_usd) * 10000:,.2f}")
        print(f"{'─'*60}\n")

    def to_dict(self) -> list[dict]:
        """Return all measurements as a list of dicts (for saving to JSON/CSV)."""
        return [
            {**asdict(m), "cost_usd": m.cost_usd, "total_tokens": m.total_tokens}
            for m in self.measurements
        ]

    def save(self, path: str):
        """Save measurements to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved {len(self.measurements)} measurements to {path}")


# -------------------------------------------------------------------
# Example / smoke test
# -------------------------------------------------------------------
if __name__ == "__main__":
    tracker = TokenTracker(model="claude-haiku-4-5-20251001")

    # --- Experiment 1: asking the LLM to do something deterministic ---
    with tracker.measure("llm_date_parse"):
        response = tracker.record(
            tracker.client.messages.create(
                model=tracker.model,
                max_tokens=64,
                messages=[{
                    "role": "user",
                    "content": "Convert '3rd of April 2026' to ISO 8601 format. Reply with only the date string."
                }]
            )
        )
        llm_result = response.content[0].text.strip()
        print(f"LLM date result: {llm_result}")

    # --- Experiment 2: doing the same thing in Python ---
    from dateutil import parser as dateparser
    import time

    with tracker.measure("python_date_parse"):
        # No API call — zero tokens, but we still record the measurement
        # so it shows up in comparisons
        t0 = time.perf_counter()
        python_result = dateparser.parse("3rd of April 2026").date().isoformat()
        # Manually set latency since we're not making an API call
        tracker._current.latency_ms = (time.perf_counter() - t0) * 1000
        print(f"Python date result: {python_result}")

    # --- Experiment 3: a legitimate LLM task ---
    with tracker.measure("llm_genuine_task"):
        response = tracker.record(
            tracker.client.messages.create(
                model=tracker.model,
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": (
                        "In 2-3 sentences, explain why using an LLM to parse dates "
                        "is wasteful compared to using a library like dateutil."
                    )
                }]
            )
        )
        print(f"\nLLM explanation:\n{response.content[0].text.strip()}\n")

    # --- Report ---
    tracker.report()
    tracker.compare("llm_date_parse", "llm_genuine_task")
    tracker.save("data/smoke_test_results.json")