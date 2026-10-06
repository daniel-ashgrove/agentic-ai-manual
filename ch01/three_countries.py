"""Chapter 1, Try It Yourself: one solution to the three-country extension.

It starts from agent_vs_chatbot.py (Section 1.5) and changes only what the exercise
asks for: a third country in the tool's data, and a reasoning step that looks up every
country it has not seen yet before it finishes. The loop in run_agent keeps its shape:
decide, act, observe, and drop a failed result and try again.

    python three_countries.py
"""
import agent_vs_chatbot as base

# Source: Eurostat, population on 1 January 2025, the same release as the other two.
base.POPULATION_DATA["italy"] = 58_934_200

COUNTRIES = ("france", "germany", "italy")


def decide_next_step(memory: dict) -> dict:
    """Stand-in for an LLM reasoning step, now for any number of countries."""
    for country in COUNTRIES:
        if country not in memory:
            return {"action": "call_tool", "arg": country}
    return {"action": "finish"}


def run_agent(max_steps: int = 8) -> str:
    memory, steps = {}, 0
    for _ in range(max_steps):
        steps += 1
        step = decide_next_step(memory)
        if step["action"] == "finish":
            break

        result = base.get_population(step["arg"])  # act
        if not result["ok"]:
            # observe + self-correct: drop the bad result, try again
            continue
        memory[step["arg"]] = result["value"]  # observe: record it

    if any(country not in memory for country in COUNTRIES):
        return "Could not complete the task within the step budget."

    ranked = sorted(COUNTRIES, key=memory.get, reverse=True)
    lines = [f"{n}. {c.title()}: {memory[c]:,}" for n, c in enumerate(ranked, 1)]
    gaps = [f"{a.title()} has {memory[a] - memory[b]:,} more than {b.title()}"
            for a, b in zip(ranked, ranked[1:])]
    return "\n".join(lines + ["; ".join(gaps) + ".", f"Resolved in {steps} steps."])


if __name__ == "__main__":
    print(run_agent())
