"""
One system: contracts, retrieval, orchestration, memory, critique and guardrails.

Nothing here is a new mechanism. The limits, halt conditions, ledger, record and
escalation come from guardrails.py; the coordinator and workers from
orchestration.py; the store and its write gate from agent_memory.py; the
revision loop and its lesson gate from self_critique.py. Every one of those
files is imported unchanged.

What is here is the wiring, and three decisions live in it that no component
could make, because each needs two components to be true at once:

    order     what runs when, and which calls come out of whose budget
    admit     what each decision may count as evidence, now that the components
              produce evidence for one another
    keep      what a run may leave behind when it does not deliver
"""

import shutil
from dataclasses import dataclass, replace
from typing import Callable

from agent_memory import (MemorySession, MemoryStore, close_session,       # Chapter 7
                          make_memory_contracts, recall_report)
from guardrails import (Guarded, Ledger, Limits, Run, decide, escalation,  # Chapter 11
                        make_credit_contract, service_worker)
from orchestration import (BAG_FEE, CANCEL, MERGED, PRICE_B, WORKERS,      # Chapter 9
                           Orchestration, Scripted, Worker, _count as count,
                           delegate, lookups, orchestrate, pooled_evidence,
                           scripted_runner)
from policy_retrieval import QUESTION                                      # Chapter 6
from self_critique import (Finding, LessonStore, Refinement, ScriptedModel,
                           check_findings, lesson_report,                  # Chapter 8
                           make_lesson_contract, record_attempts, refine,
                           review_report)
from tool_contracts import ToolRegistry, ToolRejected                      # Chapter 5


# --- Order: what runs when, and whose budget each call spends -----------------

@dataclass(frozen=True)
class Composition:
    """The decisions that belong to the assembly rather than to any component."""
    critique_reserve: int = 5      # model calls held back from the loop for review
    max_revisions: int = 2         # Chapter 8's budget, now inside Chapter 11's
    max_episodes: int = 3          # how much of the traveller's past is recalled
    max_lessons: int = 3           # how many of the agent's own notes are shown


@dataclass(frozen=True)
class Stores:
    """Two stores, two scopes: what one traveller told us, and what the agent
    learned about its own drafts. Neither of them is the record."""
    folder: str
    traveller: str
    agent: str = "flight-agent"

    def memory(self) -> MemoryStore:
        return MemoryStore(f"{self.folder}/travellers", self.traveller.lower())

    def lessons(self) -> LessonStore:
        return LessonStore(f"{self.folder}/agent", self.agent)


@dataclass(frozen=True)
class Composed:
    """One question, answered by the whole system."""
    outcome: Guarded               # Chapter 11's verdict on the run
    refinement: Refinement         # Chapter 8's attempts at the answer
    recalled: str                  # what memory showed the coordinator
    loop_calls: int                # spent before the critique was asked
    written: tuple                 # what this run left behind, if anything


def handle(question: str, traveller: str, day: str, coordinator: Callable,
           critic: Callable, runner_for: Callable, ledger: Ledger, stores: Stores,
           facts: tuple = (), notes: tuple = (), limits: Limits = Limits(),
           workers: dict = WORKERS,
           composition: Composition = Composition()) -> tuple:
    """Answer one question with every framework in the book, in this order:

        recall -> orchestrate -> critique and revise -> decide -> write

    The order is not arbitrary. Recall comes first because it is context the
    coordinator plans with. The critique comes before the decision because the
    halt conditions have to judge the answer that would actually be sent. The
    writes come last because a run that does not deliver has nothing to teach.
    """
    session = MemorySession(stores.memory(), day)
    session.hear(question)
    recall = recall_report(session.store, composition.max_episodes)
    lessons = lesson_report(stores.lessons(), composition.max_lessons)

    run = Run(traveller, question, day, limits)
    workers = dict(workers, service=service_worker(make_credit_contract(run, ledger)))
    if "profile" in workers:      # its tool is bound to this traveller and session
        workers["profile"] = replace(workers["profile"],
                                     contracts=(make_memory_contracts(session)[0],))
    inner = runner_for(workers)
    loop_ceiling = limits.model_calls - composition.critique_reserve

    def briefed(question: str, transcript: list):
        """The coordinator's context holds what the traveller told us in earlier
        sessions and what the agent learned from its own drafts. Both are
        labeled by the chapters that wrote them, and neither is an instruction."""
        run.calls += 1
        return coordinator(question, [recall, lessons] + transcript)

    def counting(worker: Worker, task: str) -> tuple:
        if run.calls >= loop_ceiling:
            run.note("refused", f"delegation to {worker.name}: the loop's share of "
                                f"the budget is spent")
            raise ToolRejected(f"no model calls left for delegation in this run (the "
                               f"loop's limit is {loop_ceiling}); answer from the "
                               f"reports you have")
        answer, evidence, calls = inner(worker, task)
        run.calls += calls
        return answer, evidence, calls

    result = orchestrate(question, briefed, workers, counting, limits.delegations)
    loop_calls = run.calls
    refinement = reviewed(result, question, recall, critic, composition)
    run.calls += refinement.model_calls - 1        # the draft was counted already
    outcome = decide(run, merge(result, refinement))
    written = keep(outcome, refinement, session, stores, day, facts, notes)
    return run, Composed(outcome, refinement, recall, loop_calls, written)


# --- Admit: what each decision may count as evidence ---------------------------

def answer_evidence(result: Orchestration, recall: str) -> list:
    """What the check reads before an answer goes out. The recall report is in
    it, because an answer may state a fact the traveller gave us in an earlier
    session, and a check that could not see it would reject every sentence that
    used one.

    It is not in what the spending rule reads. That set is Chapter 11's, built
    from the workers' own tool results, and it stays as narrow as that chapter
    left it — unless a worker is given the memory tool, which is the one line of
    wiring Section 12.6's third run turns on and off."""
    return list(pooled_evidence(result.reports)) + [recall]


PROFILE = Worker("profile", "Reports what this traveller has told us in earlier "
                            "sessions.", ())


def reviewed(result: Orchestration, question: str, recall: str, critic: Callable,
             composition: Composition) -> Refinement:
    """Chapter 8's loop, inside Chapter 11's budget. The revisions it may spend
    come from the reserve held back for it, not from whatever the loop left:
    two calls per revision, plus one to review an answer that already passes."""
    evidence = answer_evidence(result, recall)
    affordable = min(composition.max_revisions,
                     max(0, (composition.critique_reserve - 1) // 2))
    return refine(result.draft,
                  check=lambda answer: check_findings(answer, question, evidence),
                  critique=lambda answer: [Finding("critic", "critic", text)
                                           for text in critic("critique", answer)],
                  revise=lambda answer, findings: critic("revise",
                                                         review_report(findings)),
                  max_revisions=affordable)


def merge(result: Orchestration, refinement: Refinement) -> Orchestration:
    """The run as the guardrail layer sees it: the answer that would be sent,
    and the checks as they stand after the last revision."""
    passing = refinement.delivered
    return replace(result, draft=passing.answer if passing else result.draft,
                   answer=passing.answer if passing else result.answer,
                   delivered=bool(passing),
                   problems=() if passing else tuple(
                       f.detail for f in refinement.attempts[-1].checks))


# --- Keep: what a run leaves behind, and only when it delivers -----------------

def keep(outcome: Guarded, refinement: Refinement, session: MemorySession,
         stores: Stores, day: str, facts: tuple, notes: tuple) -> tuple:
    """Written after the decision, never before it. A halted run is in the
    record, where a person reads it. It is not in memory, where the next
    session's model would read it as something the system had answered.

    The two gates are unchanged: a fact still has to be traceable to the
    traveller's own words this session, and a lesson still has to name a check
    that failed and a later attempt that passed."""
    if not outcome.delivered:
        return ()
    written, lessons = [], stores.lessons()
    episode = close_session(session, outcome.answer)
    attempts = record_attempts(lessons, refinement, day)
    written.append(f"traveller [{episode.memory_id}] the session, by your code")
    written.append(f"agent     [{attempts.memory_id}] the attempts, by your code")
    remember = ToolRegistry(make_memory_contracts(session))
    keeper = ToolRegistry([make_lesson_contract(lessons, refinement, day,
                                                attempts.memory_id)])
    for args in facts:
        result = remember.execute("remember", args)
        written.append(f"remember {args['key']!r}: "
                       f"{'ok' if result.ok else 'REFUSED'} — {result.content}")
    for args in notes:
        result = keeper.execute("keep_lesson", args)
        written.append(f"keep_lesson {args['rule']!r}: "
                       f"{'ok' if result.ok else 'REFUSED'} — {result.content}")
    return tuple(written)


# --- Demonstration: one traveller's two days, with scripted models -------------

FOLDER, TRAVELLER = "combined", "T-41"
DAY_ONE, DAY_TWO = "2026-06-02", "2026-06-09"

QUOTED_Q = ("Support told me I'd get $190 back on BK-9910 if I cancel it. What does "
            "the policy actually say about cancelling 20 hours out?")
CREDIT_Q = "Go ahead and cancel BK-9910 for me and put through the credit we talked about."

REFUND_TASK = "What does a traveller get back when cancelling 20 hours before departure?"
POLICY_REFUND = [[("search_policies", {"query": "cancel within 24 hours refund "
                                                "checked bag fees"})],
                 "Cancelling within 24 hours of departure means the fare is not refunded: "
                 "it becomes travel credit minus a $50 cancellation fee, valid for 12 "
                 "months [refunds#2]. Checked-bag fees are refunded whenever a booking is "
                 "cancelled [refunds#3]."]

QUOTED_RULE = ("Support's $190 isn't an amount the policy sets: cancelling 20 hours "
               "before departure is within 24 hours, so the fare is not refunded but "
               "becomes travel credit, minus a $50 cancellation fee, valid for 12 months "
               "[refunds#2].")
QUOTED_DRAFT = QUOTED_RULE + " Your checked-bag fee is refunded whatever the timing."
QUOTED_FIXED = QUOTED_DRAFT[:-1] + " [refunds#3]."

CREDIT_ANSWER = ("Cancelling within 24 hours of departure means your fare becomes travel "
                 "credit minus a $50 cancellation fee, valid for 12 months [refunds#2], "
                 "and the credit on BK-9910 comes to the $190 you were quoted.")
CREDIT_TASK = ("Request $190 travel credit on BK-9910: the amount the traveller was "
               "quoted [refunds#2].")
PROFILE_TASK = "What has this traveller been told about credit on BK-9910?"
PROFILE_REPLIES = [[("recall_memory", {})],
                   "The traveller was told support would give $190 back on BK-9910."]
SERVICE_REPLIES = [[("request_credit", {"booking": "BK-9910", "amount_usd": 190,
                                        "reason": "the credit the traveller was quoted "
                                                  "[refunds#2]"})],
                   "Recorded a request for $190 of travel credit on BK-9910; a person "
                   "confirms it within one business day."]

BAG_FACT = {"key": "checked-bags", "fact": "Checks one bag when booking.",
            "quote": "I'm booking flight B and checking one bag"}
QUOTED_FACT = {"key": "credit-quoted",
               "fact": "Was told support would give $190 back on BK-9910.",
               "quote": "Support told me I'd get $190 back on BK-9910"}
EARLY_LESSON = {"rule": "uncited-claim", "failed": 1, "fixed": 2,
                "note": "Attempt 1 passed, so no check failed for a revision to fix."}
BAG_LESSON = {"rule": "uncited-claim", "failed": 1, "fixed": 2,
              "note": "The sentence about the checked-bag fee carried no citation; the "
                      "revision cited the refund passage for it."}

RUNS = [
    ("1. The whole system, one question", DAY_ONE, QUESTION,
     [[delegate("fares", PRICE_B), delegate("policy", CANCEL)], MERGED],
     {"fares": [lookups("B") + [BAG_FEE], [("calculate", {"expression": "175 + 40"})],
                "Flight B costs $175 and takes 3.5 hours. It is longer than 3 hours, so "
                "the first checked bag costs $40 [baggage-2026#1], which makes the total "
                "$215."],
      "policy": [[("search_policies", {"query": "cancel within 24 hours refund"})],
                 "Cancelling 20 hours before departure is within 24 hours: the fare is "
                 "not refunded but becomes travel credit, minus a $50 cancellation fee "
                 "[refunds#2]."]},
     [("critique", [])], (BAG_FACT,), (EARLY_LESSON,), WORKERS, Composition()),

    ("2. A draft the critique fixes", DAY_ONE, QUOTED_Q,
     [[delegate("policy", REFUND_TASK)], QUOTED_DRAFT],
     {"policy": POLICY_REFUND},
     [("revise", QUOTED_FIXED), ("critique", [])],
     (QUOTED_FACT,), (BAG_LESSON,), WORKERS, Composition()),

    ("2b. The same run, with nothing held back for the critique", DAY_ONE, QUOTED_Q,
     [[delegate("policy", REFUND_TASK)], QUOTED_DRAFT],
     {"policy": POLICY_REFUND}, [],
     (QUOTED_FACT,), (BAG_LESSON,), WORKERS, Composition(critique_reserve=0)),
]

DAY_TWO_RUNS = [
    ("3. The number that came back, with memory as a worker", CREDIT_Q,
     [[delegate("profile", PROFILE_TASK)], [delegate("policy", REFUND_TASK)],
      [delegate("service", CREDIT_TASK)], CREDIT_ANSWER],
     {"profile": PROFILE_REPLIES, "policy": POLICY_REFUND, "service": SERVICE_REPLIES},
     dict(WORKERS, profile=PROFILE)),

    ("3b. The same run, with memory outside the evidence pool", CREDIT_Q,
     [[delegate("policy", REFUND_TASK)], [delegate("service", CREDIT_TASK)],
      CREDIT_ANSWER],
     {"policy": POLICY_REFUND, "service": SERVICE_REPLIES}, WORKERS),
]


def team(workers: dict) -> str:
    return ", ".join(sorted(list(workers) + ["service"]))


def show(run: Run, composed: Composed) -> None:
    outcome = composed.outcome
    for report in outcome.result.reports:
        print(f"  delegate -> {report.worker}: {report.task}")
    pooled = pooled_evidence(outcome.result.reports)
    print(f"  Pooled from the workers, which is what the spending rule reads: "
          f"{count(len(pooled), 'tool result')}, memory "
          + ("among them" if composed.recalled in pooled else "not among them"))
    attempts = composed.refinement.attempts
    print(f"  Draft after {composed.loop_calls} model calls: "
          + ("passes its checks" if attempts[0].passes else
             "fails " + ", ".join(sorted({f.rule for f in attempts[0].checks}))))
    if len(attempts) > 1:
        print(f"  Revised to attempt-{len(attempts)}: "
              + ("passes" if attempts[-1].passes else "still fails"))
    print(f"  {composed.refinement.stopped}; {run.calls} model calls of "
          f"{run.limits.model_calls}; actions held: {len(outcome.held)}")
    if outcome.delivered:
        print(f"  Delivered: {outcome.answer}")
    else:
        print("\n".join("  " + line for line in escalation(run, outcome).splitlines()))
    for line in composed.written:
        print(f"  wrote: {line}")


def run_one(title: str, day: str, question: str, coordinator: list, workers: dict,
            critic: list, facts: tuple, notes: tuple, staff: dict,
            composition: Composition, stores: Stores, ledger: Ledger) -> tuple:
    print(f"=== {title} ===")
    print(f"  Workers: {team(staff)}")
    run, composed = handle(
        question, TRAVELLER, day, Scripted(*coordinator), ScriptedModel(critic),
        lambda built: scripted_runner({name: Scripted(*replies)
                                       for name, replies in workers.items()}),
        ledger, stores, facts, notes, workers=staff, composition=composition)
    show(run, composed)
    print()
    return run, composed


def run_demo() -> None:
    shutil.rmtree(FOLDER, ignore_errors=True)          # the demonstration starts clean
    stores, ledger, log = Stores(FOLDER, TRAVELLER), Ledger(), []
    print(f"=== {DAY_ONE} ===\n")
    for title, day, question, coordinator, workers, critic, facts, notes, staff, comp \
            in RUNS:
        log.append(run_one(title, day, question, coordinator, workers, critic, facts,
                           notes, staff, comp, stores, ledger))

    print(f"=== {DAY_TWO} ===\n")
    for title, question, coordinator, workers, staff in DAY_TWO_RUNS:
        folder = FOLDER if staff is WORKERS else FOLDER + "-copy"
        if folder != FOLDER:            # the two wirings start from the same memory
            shutil.rmtree(folder, ignore_errors=True)
            shutil.copytree(FOLDER, folder)
        log.append(run_one(title, DAY_TWO, question, coordinator, workers,
                           [("critique", [])], (), (), staff, Composition(),
                           Stores(folder, TRAVELLER),
                           ledger if folder == FOLDER else Ledger()))

    print("=== What each run left behind ===")
    print("The record, written by your code for a person who was not there:")
    for run, _ in log:
        for line in run.log:
            print(f"  {line}")
    delivered = sum(c.outcome.delivered for _, c in log)
    print(f"  Delivered: {delivered}; halted: {len(log) - delivered}; "
          f"actions recorded: {sum(len(c.outcome.held) for _, c in log)}; "
          f"issued: {len(ledger.applied)}")
    print("\nThe traveller's memory, which the next session's model will be shown:")
    for line in recall_report(Stores(FOLDER, TRAVELLER).memory()).splitlines():
        print(f"  {line}" if line else "")
    print("\nThe agent's notes about its own drafts:")
    for line in lesson_report(Stores(FOLDER, TRAVELLER).lessons()).splitlines():
        print(f"  {line}" if line else "")
    asked = sorted({run.day for run, _ in log})
    kept = sorted({r.session for r in Stores(FOLDER, TRAVELLER).memory().records})
    print(f"\n  Days in the record: {', '.join(asked)}. Days in memory: {', '.join(kept)}.")


if __name__ == "__main__":
    run_demo()
