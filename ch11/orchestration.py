"""
A coordinator and workers, standard library only.

The coordinator splits a traveller's question into tasks and delegates each one
to a worker agent with its own tools. Three decisions stay with your code, not
the models:

    scope     what each worker can do and see
    brief     what each worker is told, and that a brief is not evidence
    merge     what comes back, and what counts: tool results, never reports
"""

import re
import threading
from dataclasses import dataclass, replace
from typing import Callable

from policy_retrieval import (DOCUMENTS, QUESTION, PolicyLibrary,      # Chapter 6
                              check_answer, make_search_contract)
from tool_contracts import CONTRACTS, ToolContract, ToolRegistry      # Chapter 5


# --- Scope: each worker's tools and library slice, fixed by your code ---------

@dataclass(frozen=True)
class Worker:
    name: str
    role: str                  # what the coordinator is told this worker does
    contracts: tuple           # the only tools it can call


def library_slice(topic: str, prefixes: tuple) -> ToolContract:
    """Chapter 6's search over some documents only, described as exactly that."""
    documents = [d for d in DOCUMENTS if d.doc_id.startswith(prefixes)]
    return replace(make_search_contract(PolicyLibrary(documents)), description=(
        f"Search the airline's {topic} policies. Returns up to max_results "
        f"passages, each headed by an ID in square brackets and its source. Cite the "
        f"ID of every passage you rely on. If nothing matches, say these policies do "
        f"not cover it."))


WORKERS = {
    "fares": Worker("fares",
                    "Prices flights: fares, times, checked-bag fees and totals.",
                    tuple(CONTRACTS) + (library_slice("baggage", ("baggage",)),)),
    "policy": Worker("policy",
                     "Answers questions about cancellations, refunds and pets.",
                     (library_slice("cancellation, refund and pet",
                                    ("refunds", "pets")),)),
}


# --- Briefs and reports: what crosses between agents --------------------------

@dataclass(frozen=True)
class Report:
    worker: str
    task: str                  # the brief: the coordinator's words
    answer: str                # the worker's summary: a claim
    evidence: tuple            # the tool results it was shown, collected by your code
    problems: tuple            # the check, against the traveller's words and evidence
    model_calls: int

    @property
    def passes(self) -> bool:
        return not self.problems


Runner = Callable[[Worker, str], tuple]   # (worker, task) -> (answer, evidence, calls)


def _short(problem: str) -> str:
    return problem.split(": '")[0]


def for_coordinator(report: Report) -> str:
    """What the coordinator reads. A report that failed its checks is withheld."""
    if report.passes:
        return f"[report from {report.worker}] {report.answer}"
    return (f"[report from {report.worker} withheld] It did not pass the checks: "
            f"{'; '.join(_short(p) for p in report.problems)}. Delegate again with a "
            f"clearer task, or tell the traveller this part could not be answered.")


def make_delegate_contract(workers: dict, question: str, run: Runner, reports: list,
                           max_delegations: int = 4) -> ToolContract:
    """The coordinator's one tool. Gate 2 admits a delegation and counts it in one
    step, under a lock, because an agent loop may run one turn's calls at once.
    The handler runs the worker and checks its report before anyone reads it."""
    admitted, lock = [], threading.Lock()

    def admit(args: dict) -> list[str]:
        brief = (args["worker"], " ".join(args["task"].split()))
        with lock:
            if len(admitted) >= max_delegations:
                return [f"no delegations left (the limit is {max_delegations}); answer "
                        f"from the reports you have, or say what could not be answered"]
            if brief in admitted:
                return ["that worker already had this exact task; use its report"]
            admitted.append(brief)
        return []

    def delegate(worker: str, task: str) -> str:
        task = " ".join(task.split())
        answer, evidence, calls = run(workers[worker], task)
        # The traveller's words and the worker's own tool results are the evidence.
        # The brief is not: a number the coordinator wrote proves nothing.
        report = Report(worker, task, answer, tuple(evidence),
                        tuple(check_answer(answer, question, evidence)), calls)
        reports.append(report)
        return for_coordinator(report)

    return ToolContract(
        name="delegate",
        description=("Give one task to one worker and get its report back. Workers "
                     "see only the task you write, not the traveller's message, so "
                     "put every condition the task depends on into it. Workers: "
                     + "; ".join(f"{w.name}: {w.role}" for w in workers.values())),
        input_schema={
            "type": "object",
            "properties": {
                "worker": {"type": "string", "enum": sorted(workers),
                           "description": "Which worker gets the task."},
                "task": {"type": "string", "minLength": 10, "maxLength": 300,
                         "description": ("The task, with every condition it "
                                        "depends on.")},
            },
            "required": ["worker", "task"],
            "additionalProperties": False,
        },
        check=admit,
        handler=delegate,
    )


# --- The coordinator: delegate, merge, and check what is delivered ------------

WITHHELD = ("I couldn't produce an answer I can trace to the airline's policies. "
            "Please rephrase the question or contact support.")       # as in Chapter 6


@dataclass(frozen=True)
class Orchestration:
    answer: str                # what the traveller is sent
    delivered: bool
    draft: str                 # the coordinator's merged answer, before the check
    problems: tuple            # the final check's findings
    reports: tuple
    coordinator_calls: int     # graph level: the workers' own calls are not included

    @property
    def worker_calls(self) -> int:
        return sum(r.model_calls for r in self.reports)


def pooled_evidence(reports) -> list[str]:
    """The tool results of every worker run whose report passed. No report text is
    in it, and nothing from a run that failed: its inputs may not be sourced."""
    return [result for r in reports if r.passes for result in r.evidence]


def orchestrate(question: str, coordinator: Callable, workers: dict, run: Runner,
                max_delegations: int = 4, max_turns: int = 6) -> Orchestration:
    """coordinator(question, transcript) -> a list of (tool, args) calls, or its
    answer."""
    reports, transcript = [], []
    registry = ToolRegistry([make_delegate_contract(workers, question, run, reports,
                                                    max_delegations)])
    for turn in range(1, max_turns + 1):
        reply = coordinator(question, list(transcript))
        if isinstance(reply, str):
            problems = tuple(check_answer(reply, question, pooled_evidence(reports)))
            break
        transcript += [registry.execute(name, args).content for name, args in reply]
    else:
        reply, problems = "", (f"no answer within {max_turns} coordinator turns",)
    return Orchestration(reply if not problems else WITHHELD, not problems, reply,
                         problems, tuple(reports), turn)


class Scripted:
    """Stands in for a model: each call returns the next reply in the script,
    and a call the script didn't expect fails loudly instead of being guessed."""

    def __init__(self, *replies):
        self.replies = list(replies)

    def __call__(self, *context):
        if not self.replies:
            raise AssertionError("the script has no reply left for this call")
        return self.replies.pop(0)


def scripted_runner(models: dict) -> Runner:
    """Runs a worker as an agent loop would: each model call returns tool calls or
    an answer, and every tool call crosses the worker's own registry."""
    def run(worker: Worker, task: str) -> tuple:
        registry, seen, evidence, calls = (ToolRegistry(list(worker.contracts)),
                                           [], [], 0)
        while True:
            reply = models[worker.name](task, list(seen))
            calls += 1
            if isinstance(reply, str):
                return reply, evidence, calls
            for name, args in reply:
                outcome = registry.execute(name, args)
                seen.append(outcome.content)
                if outcome.ok:
                    evidence.append(outcome.content)
    return run


# --- Demonstration: three questions, with scripted coordinators and workers ---

CHEAPEST = "Which flight gets me there before 18:00 for the least money? I'm checking one bag."


def lookups(*flight_ids) -> list:
    return [("look_up_flight", {"flight_id": f}) for f in flight_ids]


def delegate(worker: str, task: str) -> tuple:
    return ("delegate", {"worker": worker, "task": task})


BAG_FEE = ("search_policies", {"query": "checked bag fee"})
PRICE_B = "Price flight B with one checked bag."
CANCEL = "What happens to a ticket cancelled 20 hours before departure?"
FARES_B = [lookups("B") + [BAG_FEE], [("calculate", {"expression": "175 + 40"})],
           "Flight B costs $175 and takes 3.5 hours. It is longer than 3 hours, so the "
           "first checked bag costs $40 [baggage-2026#1], which makes the total $215."]
POLICY_CANCEL = [[("search_policies", {"query": "cancel within 24 hours refund"})],
                 "Cancelling 20 hours before departure is within 24 hours: the fare is not "
                 "refunded but becomes travel credit, minus a $50 cancellation fee [refunds#2]."]
MERGED = ("Flight B costs $175 and takes 3.5 hours. It is longer than 3 hours, so the first "
          "checked bag costs $40 [baggage-2026#1], which makes the total $215. If you cancel "
          "20 hours before departure, that is within 24 hours: the fare is not refunded but "
          "becomes travel credit, minus a $50 cancellation fee [refunds#2].")

RUNS = [                 # (title, question, coordinator's replies, each worker's replies)
    ("1. Two parts that don't depend on each other", QUESTION,
     [[delegate("fares", PRICE_B), delegate("policy", CANCEL)], MERGED],
     {"fares": FARES_B, "policy": POLICY_CANCEL}),

    ("2. A number in the brief", QUESTION,
     [[delegate("fares", "Price flight B with one checked bag. The first checked bag "
                         "costs $35."), delegate("policy", CANCEL)],
      [delegate("fares", "Price flight B with one checked bag, taking the fee from the "
                         "baggage policy.")],
      MERGED],
     {"fares": [lookups("B"), [("calculate", {"expression": "175 + 35"})],
                "Flight B costs $175, and with the $35 checked-bag fee the total is $210.",
                *FARES_B],
      "policy": POLICY_CANCEL}),

    ("3a. A condition split from the comparison it changes", CHEAPEST,
     [[delegate("fares", "Find the cheapest flight that arrives before 18:00.")],
      [delegate("fares", PRICE_B)],
      "Flight B is the cheapest flight that arrives before 18:00: it costs $175 and arrives "
      "at 16:45. It takes 3.5 hours, so the first checked bag costs $40 [baggage-2026#1], "
      "which makes the total $215."],
     {"fares": [lookups("A", "B", "C"),
                "Flight B is the cheapest flight that arrives before 18:00: it costs $175 "
                "and arrives at 16:45. Flight A costs $210 and arrives at 17:30, and flight "
                "C arrives at 19:15.",
                *FARES_B]}),

    ("3b. The same question, with the condition kept in one brief", CHEAPEST,
     [[delegate("fares", "Find the cheapest flight that arrives before 18:00 for a "
                         "traveller checking one bag, counting the bag fee in each total.")],
      "Flight A is the cheapest flight that gets you there before 18:00 with your bag: it "
      "arrives at 17:30 and takes 2.5 hours, so the first checked bag is free "
      "[baggage-2026#1], and you will pay $210. Flight B has the lower fare, $175, but it "
      "takes 3.5 hours, so its first checked bag costs $40 [baggage-2026#1] and the total "
      "is $215."],
     {"fares": [lookups("A", "B", "C") + [BAG_FEE],
                [("calculate", {"expression": "175 + 40"})],
                "Flight A is the cheapest with the bag: it arrives at 17:30 and takes 2.5 "
                "hours, so its first checked bag is free [baggage-2026#1], and the total is "
                "$210. Flight B arrives at 16:45 and costs $175, but it takes 3.5 hours, so "
                "its first checked bag costs $40 [baggage-2026#1], which makes $215. Flight C "
                "arrives at 19:15, after 18:00."]}),
]


def _count(n: int, noun: str) -> str:
    return f"{n:,} {noun}" + ("" if n == 1 else "s")


def show(result: Orchestration) -> None:
    for r in result.reports:
        print(f"  delegate -> {r.worker}: {r.task}")
        print(f"      {_count(len(r.evidence), 'tool result')}, "
              f"{_count(r.model_calls, 'model call')}; report "
              + ("passes its checks" if r.passes else "WITHHELD from the coordinator"))
        for p in r.problems:
            print(f"      - {_short(p)}")
        if not r.passes and not check_answer(r.answer, r.task, list(r.evidence)):
            print("      (checked against the brief instead of the traveller's words, "
                  "it would have passed)")
    evidence = pooled_evidence(result.reports)
    passed = sum(r.passes for r in result.reports)
    print(f"  Final answer, checked against {_count(len(evidence), 'tool result')} pooled "
          f"from {_count(passed, 'passing worker run')}: "
          + ("passes" if result.delivered else "REJECTED"))
    print(f"  Delivered: {result.answer}")
    print(f"  Model calls: {result.coordinator_calls} by the coordinator (graph level), "
          f"{result.worker_calls} inside the workers")
    coordinator_read = sum(len(for_coordinator(r)) for r in result.reports)
    workers_read = sum(len(e) for r in result.reports for e in r.evidence)
    print(f"  The coordinator read {_count(coordinator_read, 'character')} of reports; the "
          f"workers read {_count(workers_read, 'character')} of tool results")


def run_demo() -> None:
    print("Workers, and the only tools each can call:")
    for w in WORKERS.values():
        topics = [re.search(r"airline's (.+?) policies", c.description) for c in w.contracts]
        names = [c.name + (f" ({t[1]})" if t else "") for c, t in zip(w.contracts, topics)]
        print(f"  {w.name:7} {', '.join(names)}")
    for title, question, coordinator, workers in RUNS:
        print(f"\n=== {title} ===")
        run = scripted_runner({name: Scripted(*replies) for name, replies in workers.items()})
        show(orchestrate(question, Scripted(*coordinator), WORKERS, run))


if __name__ == "__main__":
    run_demo()
