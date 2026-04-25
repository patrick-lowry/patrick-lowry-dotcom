"""
stock_experiment.py
-------------------
Compares three approaches to answering a portfolio question:

  Agent A (no tools):  LLM has no tools and no real data. It will hallucinate
                       prices from stale training data.

  Agent B (web search): LLM uses web search to find prices itself. Accurate
                        but floods the context with search results.

  Agent C (smart):     Python fetches prices and calculates values. LLM
                       receives only clean structured data and reasons over it.

Run with:
    python experiments/stock_experiment.py
"""

from dotenv import load_dotenv
load_dotenv()

import yfinance as yf
from token_tracker import TokenTracker

# -------------------------------------------------------------------
# Portfolio — change these to whatever you like
# -------------------------------------------------------------------
PORTFOLIO = {
    "AAPL": 150,
    "MSFT": 200,
    "NVDA": 50,
}

MODEL = "claude-sonnet-4-6"

# The same question is asked of all three agents
QUESTION = (
    "I hold {holdings}. "
    "What is my portfolio worth at today's prices, "
    "and should I be concerned about concentration risk?"
)


# -------------------------------------------------------------------
# Agent A — no tools: LLM has to answer from training data alone
# -------------------------------------------------------------------
def run_agent_a(tracker: TokenTracker):
    """
    No tools provided. The LLM will either hallucinate prices or refuse.
    Cheapest token-wise, but the answer will be wrong.
    """
    holdings_str = ", ".join(f"{qty} shares of {ticker}" for ticker, qty in PORTFOLIO.items())

    with tracker.measure("agent_a_no_tools"):
        response = tracker.record(
            tracker.client.messages.create(
                model=MODEL,
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": QUESTION.format(holdings=holdings_str)
                }]
            )
        )

    print(f"\nAgent A response:\n{response.content[0].text}\n")


# -------------------------------------------------------------------
# Agent B — web search: LLM searches for prices itself
# -------------------------------------------------------------------
def run_agent_b(tracker: TokenTracker):
    """
    LLM is given a web search tool and told to find prices itself.
    Accurate, but search results flood the context with noise.
    """
    holdings_str = ", ".join(f"{qty} shares of {ticker}" for ticker, qty in PORTFOLIO.items())

    with tracker.measure("agent_b_web_search"):
        response = tracker.record(
            tracker.client.messages.create(
                model=MODEL,
                max_tokens=1024,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=[{
                    "role": "user",
                    "content": QUESTION.format(holdings=holdings_str)
                }]
            )
        )

    # Show which searches the LLM ran
    print("\nAgent B tool calls:")
    for block in response.content:
        if block.type == "tool_use":
            print(f"  → searched: {block.input}")
        elif block.type == "text":
            print(f"\nAgent B response:\n{block.text}\n")


# -------------------------------------------------------------------
# Agent C — smart: Python does the work, LLM only reasons
# -------------------------------------------------------------------
def fetch_portfolio_data(portfolio: dict) -> dict:
    """
    Fetch current prices using yfinance and calculate position values.
    Deterministic, free, and takes milliseconds.
    """
    data = {}
    total_value = 0.0

    for ticker, qty in portfolio.items():
        price = yf.Ticker(ticker).fast_info["last_price"]
        value = price * qty
        total_value += value
        data[ticker] = {
            "price": round(price, 2),
            "shares": qty,
            "value": round(value, 2),
        }

    # Calculate concentration — deterministic arithmetic, free
    for ticker in data:
        data[ticker]["pct_of_portfolio"] = round(data[ticker]["value"] / total_value * 100, 1)

    data["total_value"] = round(total_value, 2)
    return data


def run_agent_c(tracker: TokenTracker):
    """
    Python fetches prices and computes everything deterministic.
    LLM receives only a tight, clean payload and reasons over it.
    No tools needed — the LLM has everything it needs in the prompt.
    """

    # Step 1: deterministic work in Python — zero tokens
    with tracker.measure("agent_c_data_fetch"):
        portfolio_data = fetch_portfolio_data(PORTFOLIO)
        print(f"\nAgent C fetched: {portfolio_data}\n")

    # Step 2: LLM does only the reasoning
    with tracker.measure("agent_c_llm_reasoning"):
        response = tracker.record(
            tracker.client.messages.create(
                model=MODEL,
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Here is my current portfolio with today's prices already calculated: "
                        f"{portfolio_data}. "
                        "Should I be concerned about concentration risk? "
                        "Keep your answer concise."
                    )
                }]
            )
        )

    print(f"Agent C response:\n{response.content[0].text}\n")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    tracker = TokenTracker(model=MODEL)

    print("=" * 60)
    print("Agent A — no tools (will hallucinate prices)")
    print("=" * 60)
    run_agent_a(tracker)

    print("=" * 60)
    print("Agent B — web search (accurate but expensive)")
    print("=" * 60)
    run_agent_b(tracker)

    print("=" * 60)
    print("Agent C — smart (Python fetches, LLM reasons)")
    print("=" * 60)
    run_agent_c(tracker)

    # Full report across all measurements
    tracker.report()

    # Pairwise comparisons
    tracker.compare("agent_a_no_tools",     "agent_b_web_search")
    tracker.compare("agent_b_web_search",   "agent_c_llm_reasoning")
    tracker.compare("agent_a_no_tools",     "agent_c_llm_reasoning")

    # Save for Quarto articles
    tracker.save("data/stock_experiment_results.json")