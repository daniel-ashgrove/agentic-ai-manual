"""
Minimal ReAct loop: Claude reasons, calls a calculator tool,
observes the result, and repeats until it has a final answer.
"""
import os
import sys

import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# The model is read from the BOOK_MODEL environment variable rather than
# written here, because model identifiers change more often than this book
# is reprinted. Current ones: platform.claude.com/docs/en/models/overview
MODEL = os.environ.get("BOOK_MODEL")
if not MODEL:
    sys.exit("BOOK_MODEL is not set. Set it to a current Claude model ID "
             "and run this again.")

CALCULATOR_TOOL = {
    "name": "calculate",
    "description": ("Evaluate a basic arithmetic expression and return the numeric "
                    "result."),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A basic arithmetic expression, e.g. '340 * 7'.",
            }
        },
        "required": ["expression"],
    },
}


def calculate(expression: str) -> float:
    """The one external tool this agent can call.

    Restricting the character set below narrows what can be expressed,
    but character filtering is not a general defense for eval() -- this
    pattern is scoped to a teaching example with a fixed, tiny grammar,
    not a template for production input handling.
    """
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        raise ValueError(f"Unsafe expression: {expression}")
    return eval(expression)


def run_react_loop(question: str, max_steps: int = 6) -> str:
    messages = [{"role": "user", "content": question}]

    for step in range(1, max_steps + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=[CALCULATOR_TOOL],
            # This chapter teaches the basic one-action-at-a-time ReAct
            # loop, so parallel tool calls are turned off deliberately.
            # A production system may allow a model response to contain
            # more than one tool_use block in the same turn, and must
            # handle every one of them -- this code intentionally doesn't,
            # because doing so here would be showing two lessons at once.
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            messages=messages,
        )

        for block in response.content:
            if block.type == "text" and block.text.strip():
                # This is Claude's visible response text, shown alongside
                # a tool call when there is one -- not a private reasoning
                # trace. See Section 3.2 for why that distinction matters.
                print(f"Model response (step {step}): {block.text.strip()}")

        if response.stop_reason != "tool_use":
            # Claude stopped for a reason other than requesting a tool --
            # normal completion, hitting max_tokens, or something else.
            # Treat anything other than a clean final answer as a failure
            # to reach one, rather than guessing at partial text.
            if response.stop_reason == "end_turn":
                # A reply can hold several blocks (a thinking block, then the
                # text) or, rarely, no text at all: join the text, then check.
                answer = "".join(
                    b.text for b in response.content if b.type == "text"
                )
                if answer.strip():
                    return answer
            return (
                f"Stopped early (stop_reason={response.stop_reason!r}) "
                f"before reaching a final answer."
            )

        tool_call = next(b for b in response.content if b.type == "tool_use")
        # Safe to assume exactly one tool_use block: disable_parallel_tool_use
        # above guarantees the response won't contain more than one.

        print(f"Action (step {step}): calculate({tool_call.input['expression']!r})")
        result = calculate(tool_call.input["expression"])
        print(f"Observation (step {step}): {result}")

        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": str(result),
            }],
        })

    return "Could not reach a final answer within the step budget."


if __name__ == "__main__":
    question = (
        "A server costs $340 a month. I need it for 7 months, and I get "
        "a 15% discount on the total. What do I actually pay?"
    )
    print(run_react_loop(question))
