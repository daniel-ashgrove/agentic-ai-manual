"""
Plan-and-Execute agent: a planner produces an ordered list of steps,
an executor carries them out one at a time, and a replanner checks
progress after each step and decides whether to continue or respond.
"""

import asyncio
import operator
import os
import sys
from typing import Annotated, List, Tuple, Union

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# The model is read from the BOOK_MODEL environment variable, for the
# same reason as in Chapter 3: model identifiers change more often than
# this book does. Current ones: platform.claude.com/docs/en/models/overview
MODEL = os.environ.get("BOOK_MODEL")
if not MODEL:
    sys.exit("BOOK_MODEL is not set. Set it to a current Claude model ID "
             "and run this again.")

llm = ChatAnthropic(model=MODEL)


# --- Tools -----------------------------------------------------------

@tool
def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression and return the numeric result."""
    # Same narrow-grammar caveat as Chapter 3, Section 3.5: this
    # character filter scopes eval() to a fixed, tiny grammar for a
    # teaching example. It is not a general input-safety pattern.
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        raise ValueError(f"Unsafe expression: {expression}")
    return eval(expression)


FLIGHTS = {
    "A": {"price_usd": 210, "duration_hours": 2.5, "arrival": "17:30"},
    "B": {"price_usd": 175, "duration_hours": 3.5, "arrival": "16:45"},
    "C": {"price_usd": 240, "duration_hours": 4.0, "arrival": "19:15"},
}


@tool
def look_up_flight(flight_id: str) -> dict:
    """Look up a flight's price (USD), duration (hours), and arrival
    time (24-hour clock) by its letter ID: 'A', 'B', or 'C'."""
    key = flight_id.strip().upper()
    if key not in FLIGHTS:
        return {"error": f"no flight with id {flight_id!r}"}
    return FLIGHTS[key]


# --- Shared state ------------------------------------------------------
# past_steps accumulates across the loop (operator.add), rather than
# being overwritten -- this is the plan's actual memory of what's
# already been done.

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple[str, str]], operator.add]
    response: str


# --- The planner ---------------------------------------------------

class Plan(BaseModel):
    """An ordered list of steps to follow to answer the objective."""
    steps: List[str] = Field(
        description="Steps to follow, in order. Each step must be a "
        "self-contained instruction -- specific enough that whoever "
        "executes it doesn't need to see any other step to understand "
        "what to do."
    )


planner_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "For the given objective, write a short, ordered plan. Do not "
        "add a step that isn't needed to reach the final answer, and "
        "do not skip one that is. The last step's result should be "
        "everything needed to answer the objective directly.",
    ),
    ("placeholder", "{messages}"),
])

# method="json_schema" uses Claude's structured outputs, which hold the reply to the
# schema. The default forces a tool call, which Claude Sonnet 5.5 and other current
# models reject.
planner = planner_prompt | llm.with_structured_output(Plan, method="json_schema")


async def plan_step(state: PlanExecute) -> dict:
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


# --- The executor --------------------------------------------------
# This is Chapter 3's loop, not a replacement for it: create_agent
# builds the same reason -> act -> observe cycle from Section 3.5,
# scoped here to a single step instead of the whole task.
#
# As of this writing, LangGraph's older create_react_agent prebuilt
# has been deprecated in favor of create_agent, which now lives in
# the separate `langchain` package rather than `langgraph.prebuilt`.
# It's worth noting this early in the book's library-based chapters:
# check LangGraph's own migration notes if you're reading this after
# the ground has shifted again.

agent_executor = create_agent(model=llm, tools=[calculate, look_up_flight])


async def execute_step(state: PlanExecute) -> dict:
    plan = state["plan"]
    if not plan:
        # The replanner's contract is: return a Response once nothing
        # is left to do, never an empty Plan. Reaching this point
        # with no steps means that contract was violated upstream --
        # fail loudly here rather than crashing on plan[0] below with
        # an unhelpful IndexError.
        raise RuntimeError(
            "execute_step was reached with no remaining plan steps."
        )
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = (
        f"You are executing one step of this plan:\n{plan_str}\n\n"
        f"Carry out only this step, using tools as needed: {task}"
    )
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].text)],
        "plan": plan[1:],
    }


# --- The replanner ---------------------------------------------------

class Response(BaseModel):
    """The final answer to give the user."""
    response: str


class Act(BaseModel):
    """The replanner's decision after seeing progress so far."""
    action: Union[Response, Plan] = Field(
        description="If the completed steps already answer the "
        "objective, use Response. Otherwise, use Plan with only the "
        "steps that still remain -- do not repeat a step that "
        "past_steps already shows as done."
    )


replanner_prompt = ChatPromptTemplate.from_template(
    "Objective: {input}\n\n"
    "Steps still remaining (not yet executed):\n{plan}\n\n"
    "Steps completed so far, with what each one returned:\n{past_steps}\n\n"
    "Decide what happens next: respond to the user if the objective "
    "is already answered, or continue with a plan of the remaining "
    "steps if it isn't."
)

replanner = replanner_prompt | llm.with_structured_output(Act, method="json_schema")


async def replan_step(state: PlanExecute) -> dict:
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    return {"plan": output.action.steps}


# --- Wiring the graph ------------------------------------------------

def should_end(state: PlanExecute) -> str:
    return END if state.get("response") else "agent"


workflow = StateGraph(PlanExecute)
workflow.add_node("planner", plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replan_step)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")
workflow.add_conditional_edges("replan", should_end, ["agent", END])

app = workflow.compile()


async def main() -> None:
    task = (
        "Three flights are on offer -- A, B, and C. Book the cheapest "
        "of the ones that arrive before 6pm, and tell me the total "
        "cost, including a $40 baggage fee only if that flight's "
        "duration is over 3 hours."
    )
    # recursion_limit bounds how many graph steps can run before
    # LangGraph stops the graph on its own -- a safety cutoff against
    # a loop that never reaches a Response, not a guarantee the task
    # will finish in fewer steps than this. 25 is LangGraph's own
    # default; it's set explicitly here so the number is visible.
    async for event in app.astream({"input": task}, {"recursion_limit": 25}):
        for node_name, node_output in event.items():
            if node_name != END:
                print(f"--- {node_name} ---")
                print(node_output)


if __name__ == "__main__":
    asyncio.run(main())
