"""
The guardrail layer around real agents. Chapter 9's coordinator and workers are
LangChain agents; the limits, halt conditions, record and escalation come from
guardrails.py unchanged. A third worker holds the one tool that asks for money.

Two shapes of the same checkpoint appear here. The layer's own records the request
and lets the run finish; LangChain's HumanInTheLoopMiddleware stops the graph and
waits. Section 11.8 compares them.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from guardrails import (HALTED, Guarded, Ledger, Limits, Run, decide,
                        make_credit_contract, service_worker)
from orchestra_agent import COORDINATOR_PROMPT, agent_runner, build_worker  # Chapter 9
from orchestration import (WITHHELD, WORKERS, Orchestration, Worker,
                           make_delegate_contract, pooled_evidence)
from policy_agent import ContractMiddleware, as_langchain_tool             # Section 6.7
from policy_retrieval import check_answer
from tool_contracts import ToolRegistry, ToolRejected

GUARDED_COORDINATOR_PROMPT = COORDINATOR_PROMPT + (
    " One worker, service, records a request for travel credit; it does not issue "
    "credit, and a person decides afterwards. Ask it only for an amount a worker has "
    "reported from a tool, and tell the traveller the amount you requested and that "
    "someone will confirm it."
)


class GuardedOrchestra:
    """Chapter 9's Orchestra with the guardrail layer around it. Build once, then
    call answer() per question: each question gets its own Run, its own budget and
    its own credit contract, so nothing leaks between travellers."""

    def __init__(self, coordinator_model, worker_models: dict, ledger: Ledger,
                 limits: Limits = Limits(), max_steps: int = 25):
        self.model, self.worker_models = coordinator_model, worker_models
        self.ledger, self.limits, self.max_steps = ledger, limits, max_steps

    def answer(self, question: str, traveller: str, day: str) -> tuple:
        run = Run(traveller, question, day, self.limits)
        contract = make_credit_contract(run, self.ledger)
        workers = dict(WORKERS, service=service_worker(contract))
        agents = {name: build_worker(self.worker_models[name], worker)
                  for name, worker in workers.items()}
        inner = agent_runner(agents)

        def counting(worker: Worker, task: str) -> tuple:
            if run.calls >= self.limits.model_calls:
                run.note("refused", f"delegation to {worker.name}: no model calls left")
                raise ToolRejected(f"no model calls left in this run (the limit is "
                                   f"{self.limits.model_calls}); answer from the "
                                   f"reports you have")
            answer, evidence, calls = inner(worker, task)
            run.calls += calls
            return answer, evidence, calls

        reports = []
        delegate = make_delegate_contract(workers, question, counting, reports,
                                          self.limits.delegations)
        coordinator = create_agent(
            model=self.model,
            tools=[as_langchain_tool(delegate)],
            system_prompt=GUARDED_COORDINATOR_PROMPT,
            middleware=[ContractMiddleware(ToolRegistry([delegate]))],
        )
        state, problems = {"messages": [HumanMessage(question)]}, ()
        try:
            for state in coordinator.stream(state, {"recursion_limit": self.max_steps},
                                            stream_mode="values"):
                pass
        except GraphRecursionError:
            problems = ("the coordinator did not finish within its step limit",)
        messages = state["messages"]
        draft = "" if problems else str(messages[-1].text)
        problems = problems or tuple(check_answer(draft, question,
                                                  pooled_evidence(reports)))
        run.calls += sum(isinstance(m, AIMessage) for m in messages)
        result = Orchestration(WITHHELD if problems else draft, not problems, draft,
                               problems, tuple(reports), run.calls)
        return run, decide(run, result)


# --- The other shape: the framework stops the graph and waits ----------------

APPROVE = {"decisions": [{"type": "approve"}]}
REJECT = {"decisions": [{"type": "reject"}]}


def paused_service(model, contract, thread: str = "review-1"):
    """The same tool behind LangChain's own checkpoint. The graph stops before the
    tool runs and returns an interrupt; the tool runs only when the graph is
    resumed with a decision. Nothing else about the worker changes."""
    agent = create_agent(
        model=model,
        tools=[as_langchain_tool(contract)],
        system_prompt=("Record the credit the brief asks for, then say what you "
                       "recorded."),
        middleware=[ContractMiddleware(ToolRegistry([contract])),
                    HumanInTheLoopMiddleware(interrupt_on={contract.name: True})],
        checkpointer=InMemorySaver(),
    )
    return agent, {"configurable": {"thread_id": thread}}


def pending(state: dict) -> list:
    """The requests a person is being asked about, as the interrupt reports them."""
    return [request for interrupt in state.get("__interrupt__", ())
            for request in interrupt.value["action_requests"]]


if __name__ == "__main__":
    import os
    import sys

    from langchain_anthropic import ChatAnthropic

    from guardrails import CANCEL_Q, escalation, review

    MODEL = os.environ.get("BOOK_MODEL")                 # a current Claude model ID
    if not MODEL:
        sys.exit("BOOK_MODEL is not set. Set it to a current Claude model ID "
                 "and run this again.")
    model, ledger = ChatAnthropic(model=MODEL), Ledger()
    orchestra = GuardedOrchestra(model, {name: model for name in
                                         list(WORKERS) + ["service"]}, ledger)
    run, outcome = orchestra.answer(CANCEL_Q, "T-41", "2026-06-02")
    print(outcome.answer)
    print(f"Model calls: {outcome.calls} of {run.limits.model_calls}")
    if outcome.delivered:
        for action in outcome.held:
            print(review(action, "dana", ledger, approve=True))
    else:
        print(escalation(run, outcome))
