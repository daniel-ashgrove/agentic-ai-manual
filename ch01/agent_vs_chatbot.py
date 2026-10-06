"""
Minimal agent loop vs. a single chatbot call.
"""

# --- Stand-in "world state" the tools read from -----------------------
# Source: Eurostat, population on 1 January 2025 (same reference date,
# same methodology for both countries, released July 2025).
POPULATION_DATA = {
    "france": 68_635_900,
    "germany": 83_577_100,
}

# Simulates one transient tool failure, so the loop has something
# real to recover from.
_CALL_COUNT = {"germany": 0}


def get_population(country: str) -> dict:
    """A 'tool' the agent can call. The first lookup for Germany
    fails, the second succeeds -- a stand-in for a flaky API."""
    key = country.lower()
    if key == "germany":
        _CALL_COUNT["germany"] += 1
        if _CALL_COUNT["germany"] == 1:
            return {"ok": False, "error": "timeout"}
    if key not in POPULATION_DATA:
        return {"ok": False, "error": f"no data for {country}"}
    return {"ok": True, "value": POPULATION_DATA[key]}


# --- Chatbot: one shot, no tools, no loop ------------------------------
def single_turn_chatbot(question: str) -> str:
    """One prompt in, one answer out. No tool access, no ability
    to check its own facts -- it answers from whatever it already
    'knows,' which may be wrong or out of date."""
    stale_knowledge = {"france": 65_000_000, "germany": 83_000_000}
    fr, de = stale_knowledge["france"], stale_knowledge["germany"]
    larger = "Germany" if de > fr else "France"
    hi, lo = max(fr, de), min(fr, de)
    return f"{larger} has more people (approximately {hi:,} vs {lo:,})."


# --- Agent: reason -> act -> observe, repeat ---------------------------
def decide_next_step(memory: dict) -> dict:
    """Stand-in for an LLM reasoning step: given what's been learned
    so far, decide the next action."""
    if "france" not in memory:
        return {"action": "call_tool", "arg": "france"}
    if "germany" not in memory:
        return {"action": "call_tool", "arg": "germany"}
    return {"action": "finish"}


def run_agent(max_steps: int = 6) -> str:
    memory, steps = {}, 0
    for _ in range(max_steps):
        steps += 1
        step = decide_next_step(memory)
        if step["action"] == "finish":
            break

        result = get_population(step["arg"])  # act
        if not result["ok"]:
            # observe + self-correct: drop the bad result, try again
            continue
        memory[step["arg"]] = result["value"]  # observe: record it

    fr, de = memory.get("france"), memory.get("germany")
    if fr is None or de is None:
        return "Could not complete the task within the step budget."

    diff = abs(fr - de)
    larger = "Germany" if de > fr else "France"
    return (
        f"{larger} has more people (France: {fr:,}, Germany: {de:,}, "
        f"difference: {diff:,}). Resolved in {steps} steps, "
        f"including one recovered tool failure."
    )


if __name__ == "__main__":
    print("=== Chatbot (single turn, no tools) ===")
    print(single_turn_chatbot("Which has more people, France or Germany?"))
    print()
    print("=== Agent (reason -> act -> observe loop) ===")
    print(run_agent())
