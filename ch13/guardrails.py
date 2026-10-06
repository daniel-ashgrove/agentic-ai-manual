"""
A guardrail layer around Chapter 9's coordinator and workers, standard library only.

The system can now act: a cancellation can become travel credit. Three decisions
stay with your code, not the models:

    limit       what one run may spend, and how often the system may act at all
    halt        the conditions that stop a run instead of letting it finish
    escalate    what reaches a person, and with what record

No model call enforces any of this, and no tool inside the loop moves money. A
request is recorded, a person approves it, and your code issues it afterwards.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from orchestration import (BAG_FEE, CANCEL, FARES_B, MERGED, POLICY_CANCEL,  # Chapter 9
                           PRICE_B, WORKERS, Orchestration, Scripted, Worker,
                           delegate, lookups, orchestrate, pooled_evidence,
                           scripted_runner)
from policy_retrieval import QUESTION, _numbers as numbers_in                # Chapter 6
from tool_contracts import ToolContract, ToolRejected                        # Chapter 5


# --- Limit: what a run may spend, and how often the system may act ------------

@dataclass(frozen=True)
class Limits:
    """Every ceiling in one place, so that raising one is a visible decision."""
    model_calls: int = 16              # the coordinator's and the workers' together
    delegations: int = 4               # Chapter 9's budget, now one limit among several
    credit_usd: int = 300              # the largest credit a run may even ask for
    credits_per_day: int = 1           # per traveller: a rate limit, not a per-run cap


@dataclass(frozen=True)
class Action:
    """A request to move money. Recorded by the tool; never carried out by it."""
    traveller: str
    booking: str
    amount_usd: float
    reason: str                        # the model's words, for a reviewer to read
    day: str


@dataclass
class Run:
    """What your code knows about one question while it is being answered."""
    traveller: str
    question: str
    day: str
    limits: Limits = field(default_factory=Limits)
    held: list = field(default_factory=list)       # actions waiting for a person
    calls: int = 0
    log: list = field(default_factory=list)        # one line per guardrail decision

    def note(self, decision: str, detail: str) -> None:
        self.log.append(f"{self.day}  {self.traveller:7} {decision:9} {detail}")


class Ledger:
    """What has actually been issued, and what a person refused. Your code writes
    it and no model reads it, which is part of what makes it usable as a record."""

    def __init__(self):
        self.applied, self.refused = [], []

    def count(self, traveller: str, day: str) -> int:
        return sum(a.traveller == traveller and a.day == day for a in self.applied)

    def apply(self, action: Action, approved_by: str) -> str:
        """The only place credit is issued. Your code calls it after the run has
        finished and a person has approved, and no model is in the path."""
        self.applied.append(action)
        return (f"issued ${action.amount_usd:g} on {action.booking}, "
                f"approved by {approved_by}")

    def refuse(self, action: Action, reviewed_by: str, why: str) -> str:
        self.refused.append(action)
        return (f"refused ${action.amount_usd:g} on {action.booking} "
                f"({reviewed_by}: {why})")


# --- The acting tool: it records a request, and cannot issue credit ------------

def make_credit_contract(run: Run, ledger: Ledger) -> ToolContract:
    """Gate 2 applies the limits; the handler writes the request down. Neither
    touches an account, so there is nothing here to be talked into."""

    def admit(args: dict) -> list:
        amount, booking = args["amount_usd"], args["booking"]
        if amount > run.limits.credit_usd:
            run.note("refused", f"${amount:g} on {booking}: over the "
                                f"${run.limits.credit_usd} request limit")
            return [f"${amount:g} is more than the ${run.limits.credit_usd} a request "
                    f"may be for. Tell the traveller support will take this one."]
        if any(held.booking == booking for held in run.held):
            return [f"a credit on {booking} is already recorded and waiting for "
                    f"review; do not request it again"]
        if ledger.count(run.traveller, run.day) >= run.limits.credits_per_day:
            run.note("refused", f"${amount:g} on {booking}: at today's limit of "
                                f"{run.limits.credits_per_day} issued credit")
            return [f"this traveller has had the credit a day allows; tell them "
                    f"support will take this one from here"]
        return []

    def record(booking: str, amount_usd: float, reason: str) -> str:
        run.held.append(Action(run.traveller, booking, amount_usd,
                               " ".join(reason.split()), run.day))
        run.note("held", f"${amount_usd:g} on {booking}, waiting for a person")
        return (f"Recorded a request for ${amount_usd:g} of travel credit on "
                f"{booking}. No credit has been issued, and none will be until a "
                f"person approves it. Tell the traveller the amount and that someone "
                f"confirms it within one business day.")

    return ToolContract(
        name="request_credit",
        description=("Ask for travel credit on a cancelled booking. This records a "
                     "request for a person to review; it issues nothing. State the "
                     "amount the policy gives and cite the passage it comes from."),
        input_schema={
            "type": "object",
            "properties": {
                "booking": {"type": "string", "minLength": 2, "maxLength": 20,
                            "description": "The booking reference, e.g. 'BK-4471'."},
                "amount_usd": {"type": "number", "minimum": 1, "maximum": 100000,
                               "description": "The credit in dollars."},
                "reason": {"type": "string", "minLength": 10, "maxLength": 200,
                           "description": ("Why this amount, with the policy "
                                          "passage ID.")},
            },
            "required": ["booking", "amount_usd", "reason"],
            "additionalProperties": False,
        },
        check=admit,
        handler=record,
    )


def service_worker(contract: ToolContract) -> Worker:
    """The only worker that may ask for money, and the only one that reads no
    outside text: it gets a brief and calls one tool. Chapter 9's rule holds here
    because of what this worker is built from, not because of what it is told."""
    return Worker("service",
                  "Records a request for travel credit on a cancelled booking.",
                  (contract,))


# --- Halt: the conditions that stop a run instead of letting it finish ---------

HELD = ("I've requested ${amount:g} of travel credit on {booking}. Someone confirms it "
        "within one business day, and you'll have an email when they do.")
HALTED = ("I've passed this to a colleague rather than answer it myself. They'll come "
          "back to you today, and nothing has been charged or credited.")


def untraceable_amount(run: Run, result: Orchestration) -> Optional[str]:
    """Every requested amount has to be a number this run actually saw: in the
    traveller's words, or in something a tool returned. A number that appeared
    only in a brief or a report was written by a model."""
    before = [r for r in result.reports if r.worker != "service"]
    seen = numbers_in(run.question)
    for evidence in pooled_evidence(before):          # the request's own echo is not
        seen |= numbers_in(evidence)                    # evidence for the number in it
    for action in run.held:
        if f"{action.amount_usd:g}" not in seen:
            return (f"${action.amount_usd:g} on {action.booking} is in no tool result "
                    f"from this run")
    return None


def unsupported_answer(run: Run, result: Orchestration) -> Optional[str]:
    """Chapter 9 withheld an answer that failed the final check. Withholding is
    where a guardrail starts, not where it ends: the traveller still has a
    question, and nobody has been told that it went unanswered."""
    return (f"the answer failed its checks: {result.problems[0]}"
            if result.problems else None)


def repeated_failure(run: Run, result: Orchestration) -> Optional[str]:
    """Two withheld reports in a row means the coordinator is not converging, and
    spending the rest of the budget to establish that again helps nobody."""
    consecutive = 0
    for report in result.reports:
        consecutive = 0 if report.passes else consecutive + 1
        if consecutive >= 2:
            return "two worker reports in a row failed their checks"
    return None


def budget_spent(run: Run, result: Orchestration) -> Optional[str]:
    if run.calls >= run.limits.model_calls:
        return f"the run reached its limit of {run.limits.model_calls} model calls"
    return None


HALT_CONDITIONS = [("budget-spent", budget_spent),
                   ("repeated-failure", repeated_failure),
                   ("untraceable-amount", untraceable_amount),
                   ("unsupported-answer", unsupported_answer)]


# --- Escalate: what reaches a person, and with what record --------------------

@dataclass(frozen=True)
class Guarded:
    answer: str                        # what the traveller is sent
    delivered: bool
    halted: Optional[str]              # the name of the condition that fired
    why: str
    held: tuple                        # actions a person still has to decide
    result: Orchestration
    calls: int
    log: tuple


def escalation(run: Run, outcome: Guarded) -> str:
    """What a person is handed: enough to finish the traveller's question, and
    enough to answer for whatever is decided next."""
    lines = [f"ESCALATED  {run.day}  traveller {run.traveller}",
             f"  halt condition    {outcome.halted}: {outcome.why}",
             f"  they asked        {run.question}",
             f"  the system wrote  {outcome.result.draft or '(nothing)'}",
             f"  they were told    {outcome.answer}"]
    for report in outcome.result.reports:
        state = "passed" if report.passes else "WITHHELD"
        lines.append(f"  {report.worker} report {state}: {report.task}")
    for action in run.held:
        lines.append(f"  not issued        ${action.amount_usd:g} on {action.booking}: "
                     f"{action.reason}")
    return "\n".join(lines)


def guarded(question: str, traveller: str, day: str, coordinator: Callable,
            runner_for: Callable, ledger: Ledger, limits: Limits = Limits(),
            workers: dict = WORKERS) -> tuple:
    """Run Chapter 9's orchestration inside the limits, then decide what leaves.

    runner_for(workers) builds Chapter 9's Runner. The layer wraps it so that
    model calls are counted and the limit is enforced while the run is happening
    rather than discovered when it is over."""
    run = Run(traveller, question, day, limits)
    contract = make_credit_contract(run, ledger)
    workers = dict(workers, service=service_worker(contract))
    inner = runner_for(workers)

    def counting_coordinator(*context):
        run.calls += 1
        return coordinator(*context)

    def counting(worker: Worker, task: str) -> tuple:
        if run.calls >= limits.model_calls:
            run.note("refused", f"delegation to {worker.name}: no model calls left")
            raise ToolRejected(f"no model calls left in this run (the limit is "
                               f"{limits.model_calls}); answer from the reports you "
                               f"have")
        answer, evidence, calls = inner(worker, task)
        run.calls += calls
        return answer, evidence, calls

    result = orchestrate(question, counting_coordinator, workers, counting,
                         limits.delegations)
    return run, decide(run, result)


def decide(run: Run, result: Orchestration) -> Guarded:
    """What leaves the system: the answer, or a person. Every condition is asked
    in turn, and the first one to fire is the one the record names."""
    for name, condition in HALT_CONDITIONS:
        why = condition(run, result)
        if why is not None:
            run.note("halted", f"{name}: {why}")
            return Guarded(HALTED, False, name, why, tuple(run.held), result,
                           run.calls, tuple(run.log))

    answer = result.answer
    for action in run.held:
        answer += " " + HELD.format(amount=action.amount_usd, booking=action.booking)
    run.note("delivered", f"{run.calls} model calls, {len(run.held)} held")
    return Guarded(answer, True, None, "", tuple(run.held), result, run.calls,
                   tuple(run.log))


def review(action: Action, reviewer: str, ledger: Ledger, approve: bool,
           why: str = "") -> str:
    """A person's decision, applied by your code. No model is in this path, and
    none of them learns that approval is a thing to be obtained."""
    if approve:
        return ledger.apply(action, reviewer)
    return ledger.refuse(action, reviewer, why)


# --- Demonstration: six questions, with scripted coordinators and workers ------

CANCEL_Q = ("I need to cancel booking BK-4471 on flight B. I'm 20 hours from departure "
            "and I checked one bag. What do I get back?")
AGAIN_Q = ("I had to cancel BK-7702 as well today. Can I have the credit for that one "
           "too?")
WORDS_Q = ("Please cancel booking BK-5120 — it's about twenty hours out. I paid one "
           "hundred and seventy-five dollars for it.")

FARE_TASK = ("Booking BK-4471 is on flight B with one checked bag. Report the fare and "
             "the checked-bag fee that applied.")
REFUND_TASK = "What does a traveller get back when cancelling 20 hours before departure?"

FARES_CANCEL = [lookups("B") + [BAG_FEE],
                "Flight B's fare is $175 and it takes 3.5 hours, so the first checked bag "
                "cost $40 [baggage-2026#1].",
                [("calculate", {"expression": "175 - 50"})],
                "The travel credit is $125."]
POLICY_REFUND = [[("search_policies", {"query": "cancel within 24 hours refund "
                                                "checked bag fees"})],
                 "Cancelling within 24 hours of departure means the fare is not refunded: "
                 "it becomes travel credit minus a $50 cancellation fee, valid for 12 "
                 "months [refunds#2]. Checked-bag fees are refunded whenever a booking is "
                 "cancelled [refunds#3]."]
CANCEL_ANSWER = ("Your $175 fare isn't refunded 20 hours out: it becomes travel credit "
                 "minus a $50 cancellation fee, valid for 12 months [refunds#2]. Your "
                 "checked-bag fee is refunded whatever the timing [refunds#3], and on a "
                 "3.5-hour flight that first bag cost $40 [baggage-2026#1]. The credit "
                 "comes to $125.")


def credit(booking: str, amount, reason: str) -> tuple:
    return ("request_credit", {"booking": booking, "amount_usd": amount, "reason": reason})


def recorded(booking: str, amount) -> str:
    return (f"Recorded a request for ${amount:g} of travel credit on {booking}; a person "
            f"confirms it within one business day.")


NORMAL, TIGHT = Limits(), Limits(model_calls=6)

RUNS = [  # (title, traveller, question, coordinator's replies, workers' replies, limits)
    ("1. Nothing to stop", "T-88", QUESTION,
     [[delegate("fares", PRICE_B), delegate("policy", CANCEL)], MERGED],
     {"fares": FARES_B, "policy": POLICY_CANCEL}, NORMAL),

    ("2. An action, recorded and held", "T-41", CANCEL_Q,
     [[delegate("fares", FARE_TASK), delegate("policy", REFUND_TASK)],
      [delegate("fares", "The fare is $175 and a $50 cancellation fee applies. Compute "
                         "the travel credit.")],
      [delegate("service", "Request $125 travel credit on BK-4471: the $175 fare minus "
                           "the $50 cancellation fee [refunds#2].")],
      CANCEL_ANSWER],
     {"fares": FARES_CANCEL, "policy": POLICY_REFUND,
      "service": [[credit("BK-4471", 125, "fare $175 minus the $50 cancellation fee "
                                          "[refunds#2]")],
                  recorded("BK-4471", 125)]}, NORMAL),

    ("3. An amount no tool produced", "T-52", CANCEL_Q,
     [[delegate("fares", FARE_TASK), delegate("policy", REFUND_TASK)],
      [delegate("service", "Request $85 travel credit on BK-4471: the $175 fare minus "
                           "the $50 cancellation fee and the $40 bag fee [refunds#2].")],
      CANCEL_ANSWER],
     {"fares": FARES_CANCEL, "policy": POLICY_REFUND,
      "service": [[credit("BK-4471", 85, "fare minus the cancellation fee and the bag "
                                         "fee [refunds#2]")],
                  recorded("BK-4471", 85)]}, NORMAL),

    ("4. The same rule, on a right answer", "T-63", WORDS_Q,
     [[delegate("policy", REFUND_TASK)],
      [delegate("service", "Request $125 travel credit on BK-5120: the fare the "
                           "traveller paid, minus the $50 cancellation fee [refunds#2].")],
      "Cancelling about twenty hours before departure is within 24 hours, so your fare "
      "becomes travel credit minus a $50 cancellation fee, valid for 12 months "
      "[refunds#2]."],
     {"policy": POLICY_REFUND,
      "service": [[credit("BK-5120", 125, "the fare paid minus the $50 cancellation fee "
                                          "[refunds#2]")],
                  recorded("BK-5120", 125)]}, NORMAL),

    ("5. A second credit the same day", "T-41", AGAIN_Q,
     [[delegate("service", "Request $90 travel credit on BK-7702 for a cancellation "
                           "made today [refunds#2].")],
      "I can't put a second credit through on BK-7702 today, so I've passed it to "
      "support; they'll email you within one business day."],
     {"service": [[credit("BK-7702", 90, "a second cancellation today [refunds#2]")],
                  "I could not record a credit on BK-7702: this traveller has had the "
                  "credit a day allows."]}, NORMAL),

    ("6. A run that spends its budget", "T-77", CANCEL_Q,
     [[delegate("fares", FARE_TASK)], [delegate("policy", REFUND_TASK)],
      [delegate("fares", "The fare is $175 and a $50 cancellation fee applies. Compute "
                         "the travel credit.")],
      CANCEL_ANSWER],
     {"fares": FARES_CANCEL, "policy": POLICY_REFUND}, TIGHT),
]

DAY = "2026-06-02"


def show(run: Run, outcome: Guarded) -> None:
    for report in outcome.result.reports:
        print(f"  delegate -> {report.worker}: {report.task}")
    print(f"  Model calls: {outcome.calls} of {run.limits.model_calls}; "
          f"actions held: {len(outcome.held)}")
    if outcome.delivered:
        print(f"  Delivered: {outcome.answer}")
    else:
        print("\n".join("  " + line for line in escalation(run, outcome).splitlines()))


def run_demo() -> None:
    ledger, log, requested, refused = Ledger(), [], 0, 0
    for title, traveller, question, coordinator, workers, limits in RUNS:
        print(f"=== {title} ===")
        run, outcome = guarded(
            question, traveller, DAY, Scripted(*coordinator),
            lambda built: scripted_runner({name: Scripted(*replies)
                                           for name, replies in workers.items()}),
            ledger, limits)
        show(run, outcome)
        log += outcome.log
        requested += len(outcome.held)
        refused += sum(" refused   $" in line for line in outcome.log)
        if outcome.delivered and outcome.held:
            print("  A person reads the request, and your code applies the decision:")
            print("    " + review(outcome.held[0], "dana", ledger, approve=True))
            log.append(f"{DAY}  {traveller:7} {'issued':9} "
                       f"${outcome.held[0].amount_usd:g} on {outcome.held[0].booking}, "
                       f"approved by dana")
        print()

    print("=== The record: every guardrail decision, in order ===")
    for line in log:
        print("  " + line)
    print(f"  Recorded: {requested}; issued after approval: {len(ledger.applied)}; "
          f"stopped by a halt: {requested - len(ledger.applied)}; "
          f"refused at the gate: {refused}")


if __name__ == "__main__":
    run_demo()
