"""
weather_experiment.py
---------------------
Compares three architectures for answering "what's the weather outlook
for a walk near me?" in a hypothetical walking/running route app.

  Agent A: LLM only — no tools, no data. Will hallucinate.

  Agent B: LLM with tools — two tools provided for finding nearby cities
           and looking up weather. LLM orchestrates everything.

  Agent C: Smart — Python fetches nearby cities and weather, summarises
           the data, LLM only writes the natural language recommendation.

APIs used (both free, no key required):
  - Nominatim (OpenStreetMap) for nearby city lookup
  - Open-Meteo for weather forecasts

Run with:
    python experiments/weather_experiment.py
"""

from dotenv import load_dotenv
load_dotenv()

import json
import time
import requests
from token_tracker import TokenTracker

# -------------------------------------------------------------------
# Test user location — central London
# -------------------------------------------------------------------
USER_LOCATION = {
    "lat": 51.5074,
    "lon": -0.1278,
    "description": "Central London"
}

RADIUS_KM = 30
MODEL = "claude-sonnet-4-6"

# The same user prompt for all three agents
USER_PROMPT = (
    f"What's the weather outlook for a walk near me? "
    f"I'm in {USER_LOCATION['description']}."
)


# -------------------------------------------------------------------
# Deterministic helper functions (used by Agent C, and exposed as
# tool implementations for Agent B)
# -------------------------------------------------------------------

def get_nearby_cities(lat: float, lon: float, radius_km: int = RADIUS_KM) -> list[dict]:
    """
    Find towns and cities within radius_km of a coordinate using Nominatim.
    Returns a list of dicts with name, lat, lon.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": "city",
        "format": "json",
        "limit": 8,
        "viewbox": (
            f"{lon - radius_km/69},{lat + radius_km/69},"
            f"{lon + radius_km/69},{lat - radius_km/69}"
        ),
        "bounded": 1,
        "featuretype": "city"
    }
    headers = {"User-Agent": "weather-experiment/1.0"}
    response = requests.get(url, params=params, headers=headers)
    results = response.json()

    cities = []
    for r in results:
        cities.append({
            "name": r["display_name"].split(",")[0],
            "lat": float(r["lat"]),
            "lon": float(r["lon"])
        })

    # Always include the user's own location
    cities.insert(0, {"name": USER_LOCATION["description"], "lat": lat, "lon": lon})
    return cities[:6]  # cap at 6 to keep context manageable


def get_weather(lat: float, lon: float) -> dict:
    """
    Fetch current weather and today's forecast from Open-Meteo.
    Returns a clean dict of the most relevant fields.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,weathercode,windspeed_10m,precipitation",
        "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "Europe/London",
        "forecast_days": 1
    }
    response = requests.get(url, params=params)
    data = response.json()

    return {
        "temperature_c": data["current"]["temperature_2m"],
        "wind_kph": data["current"]["windspeed_10m"],
        "precipitation_mm": data["current"]["precipitation"],
        "weather_code": data["current"]["weathercode"],
        "max_temp_c": data["daily"]["temperature_2m_max"][0],
        "min_temp_c": data["daily"]["temperature_2m_min"][0],
        "daily_precipitation_mm": data["daily"]["precipitation_sum"][0],
    }


# WMO weather codes -> human readable (subset)
WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "icy fog", 51: "light drizzle", 53: "moderate drizzle",
    61: "light rain", 63: "moderate rain", 65: "heavy rain",
    71: "light snow", 73: "moderate snow", 80: "light showers",
    81: "moderate showers", 82: "heavy showers", 95: "thunderstorm"
}


# -------------------------------------------------------------------
# Agent A — no tools, no data
# -------------------------------------------------------------------
def run_agent_a(tracker: TokenTracker):
    """
    LLM receives only the user's prompt and location name.
    No real weather data. Will hallucinate or refuse.
    """
    with tracker.measure("agent_a_no_tools"):
        response = tracker.record(
            tracker.client.messages.create(
                model=MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": USER_PROMPT}]
            )
        )

    print(f"\nAgent A response:\n{response.content[0].text}\n")


# -------------------------------------------------------------------
# Agent B — LLM with two tools
# -------------------------------------------------------------------

# Tool definitions passed to the API
TOOLS = [
    {
        "name": "get_nearby_cities",
        "description": "Find towns and cities within 30km of a latitude/longitude coordinate.",
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Latitude"},
                "lon": {"type": "number", "description": "Longitude"}
            },
            "required": ["lat", "lon"]
        }
    },
    {
        "name": "get_weather",
        "description": "Get the current weather and today's forecast for a latitude/longitude.",
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Latitude"},
                "lon": {"type": "number", "description": "Longitude"}
            },
            "required": ["lat", "lon"]
        }
    }
]


def handle_tool_call(tool_name: str, tool_input: dict) -> str:
    """Dispatch tool calls to the correct Python function."""
    if tool_name == "get_nearby_cities":
        result = get_nearby_cities(tool_input["lat"], tool_input["lon"])
    elif tool_name == "get_weather":
        result = get_weather(tool_input["lat"], tool_input["lon"])
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
    return json.dumps(result)


def run_agent_b(tracker: TokenTracker):
    """
    LLM is given two tools and orchestrates the entire workflow itself.
    Each tool result is fed back into the context, growing input tokens.
    """
    messages = [{
        "role": "user",
        "content": (
            f"{USER_PROMPT} "
            f"My coordinates are {USER_LOCATION['lat']}, {USER_LOCATION['lon']}. "
            "Use the tools to find nearby cities and their weather, then give me "
            "a friendly summary and walking recommendation."
        )
    }]

    with tracker.measure("agent_b_with_tools"):
        # Agentic loop — keep going until LLM stops calling tools
        while True:
            response = tracker.record(
                tracker.client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    tools=TOOLS,
                    messages=messages
                )
            )

            # If no tool calls, we're done
            if response.stop_reason == "end_turn":
                break

            # Process tool calls and feed results back
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  Agent B called: {block.name}({block.input})")
                    result = handle_tool_call(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            # Append assistant response and tool results to history
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            if not tool_results:
                break

    # Print final text response
    for block in response.content:
        if block.type == "text":
            print(f"\nAgent B response:\n{block.text}\n")


# -------------------------------------------------------------------
# Agent C — smart: Python does everything deterministic
# -------------------------------------------------------------------
def run_agent_c(tracker: TokenTracker):
    """
    Python fetches nearby cities and weather, computes a clean summary.
    LLM receives only structured data and writes the recommendation.
    """

    # Deterministic lookups in Python — no tokens, not measured
    cities = get_nearby_cities(USER_LOCATION["lat"], USER_LOCATION["lon"])
    weather_data = []
    for city in cities:
        weather = get_weather(city["lat"], city["lon"])
        weather["city"] = city["name"]
        weather["conditions"] = WMO_CODES.get(weather["weather_code"], "unknown")
        weather_data.append(weather)
        time.sleep(0.3)  # be polite to the free API

    print(f"\nAgent C fetched weather for: {[w['city'] for w in weather_data]}\n")

    # LLM writes natural language from clean structured data
    with tracker.measure("agent_c_llm_recommendation"):
        response = tracker.record(
            tracker.client.messages.create(
                model=MODEL,
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Here is today's weather data for towns near {USER_LOCATION['description']}: "
                        f"{json.dumps(weather_data, indent=2)}. "
                        "Write a friendly 2-3 sentence weather outlook and walking recommendation "
                        "for someone planning a walk today. Do not mention specific coordinates."
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
    print("Agent A — no tools (will hallucinate)")
    print("=" * 60)
    run_agent_a(tracker)

    print("=" * 60)
    print("Agent B — LLM orchestrates tool calls")
    print("=" * 60)
    run_agent_b(tracker)

    print("=" * 60)
    print("Agent C — Python fetches data, LLM recommends")
    print("=" * 60)
    run_agent_c(tracker)

    # Full token report
    tracker.report()

    # Pairwise comparisons
    tracker.compare("agent_a_no_tools",       "agent_b_with_tools")
    tracker.compare("agent_b_with_tools",     "agent_c_llm_recommendation")
    tracker.compare("agent_a_no_tools",       "agent_c_llm_recommendation")

    # Save for Quarto
    tracker.save("data/weather_experiment_results.json")