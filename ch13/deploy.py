"""
Chapter 12's system, served as a process instead of run as a script.

Nothing about the system changes here. combined.py, guardrails.py and the six
files under them are imported exactly as their chapters left them. What is new
is everything that stops being true the moment the program stops being a
script, and three decisions live in that:

    serve   what one run is when many are in flight: its own identifier, its
            own budget, its own stores and its own deadline, and what is
            shared on purpose
    emit    what a run writes down while it happens, in a form another program
            can read: one line per decision, correlated by a run id, naming
            the release it ran under
    watch   which questions those numbers have to answer, and what a person
            does when the answer is bad (dashboard.py)
"""

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

from combined import Composition, Stores, handle                       # Chapter 12
from guardrails import Ledger, Limits, escalation, review               # Chapter 11
from tool_contracts import ToolRejected                                 # Chapter 5


# --- Serve: what one run is when many are in flight ---------------------------

@dataclass(frozen=True)
class Release:
    """What this process is running. Every event carries it, so that a change
    in behaviour can be attributed to a change in the system rather than
    argued about."""
    service: str = "flight-agent"
    version: str = "1.0.0"
    model: str = "scripted-stand-in"
    provider: str = "none.scripted"
    prompt_revision: str = "2026-06-09"


@dataclass(frozen=True)
class Request:
    question: str
    traveller: str
    day: str


@dataclass(frozen=True)
class Reply:
    """What the caller gets back. The run id is in it because it is the only
    thing that connects a traveller's complaint to the record of what
    happened."""
    run: str
    answer: str
    delivered: bool
    halted: Optional[str]
    calls: int
    held: tuple = ()


SORRY = ("Something went wrong at our end and I couldn't answer that. Nothing has "
         "been charged or credited. Please try again, or contact support.")


@dataclass(frozen=True)
class Deadline:
    """When this request stops being worth finishing.

    A deadline cannot interrupt a model call that is already in flight. What it
    can do is refuse to start the next one, which is the difference between a
    request that answers late and one that never answers at all.
    """
    at: float
    now: Callable
    refused: Callable                  # called with the worker that was turned away

    def passed(self) -> bool:
        return self.now() >= self.at

    def guard(self, inner: Callable) -> Callable:
        """Wraps whatever runner an engine builds, so the check happens at the
        one place a run can actually be stopped: between delegations."""
        def bounded(worker, task: str) -> tuple:
            if self.passed():
                self.refused(worker.name)
                raise ToolRejected("this request is out of time; answer from the "
                                   "reports you have, or say what could not be "
                                   "answered")
            return inner(worker, task)
        return bounded


class Service:
    """One process, many runs.

    Everything a run must not share is built per request: its identifier, its
    Run and budget, its credit contract, its memory session, its deadline.
    Everything that is shared is shared deliberately — the ledger, because a
    rate limit that reset every request would not be a rate limit; the lesson
    store, because it belongs to the agent rather than to a traveller; the
    event log, because it is the thing that has to hold every run at once.
    """

    def __init__(self, folder: str, log: "EventLog", ledger: Ledger, engine: Callable,
                 limits: Limits = Limits(), composition: Composition = Composition(),
                 deadline_s: float = 20.0, ids: Callable = None,
                 now: Callable = time.time):
        self.folder, self.log, self.ledger, self.engine = folder, log, ledger, engine
        self.limits, self.composition = limits, composition
        self.deadline_s, self.now = deadline_s, now
        self.ids = ids or (lambda: uuid.uuid4().hex[:12])
        self.waiting, self._lock = {}, threading.Lock()

    def answer(self, request: Request) -> Reply:
        """Answer one question. A run that fails takes nothing else down with
        it, and leaves the reason in the log rather than in a traceback nobody
        reads."""
        run_id, started = self.ids(), self.now()
        self.log.emit(run_id, "received", traveller=request.traveller, day=request.day,
                      chars=len(request.question))
        try:
            run, composed = self.engine(
                request, Stores(self.folder, request.traveller), self.ledger,
                self.limits, self.composition,
                self.deadline(run_id, started + self.deadline_s))
        except Exception as failure:
            self.log.emit(run_id, "failed", error=type(failure).__name__)
            return Reply(run_id, SORRY, False, "service-error", 0)

        outcome = composed.outcome
        for action in outcome.held:
            self.log.emit(run_id, "held", booking=action.booking,
                          amount_usd=action.amount_usd)
        self.log.emit(run_id, "answered", traveller=request.traveller,
                      delivered=outcome.delivered, halted=outcome.halted,
                      calls=run.calls, ceiling=self.limits.model_calls,
                      attempts=len(composed.refinement.attempts),
                      held=len(outcome.held), wrote=len(composed.written),
                      seconds=round(self.now() - started, 3), record=list(run.log))
        if not outcome.delivered:
            self.log.emit(run_id, "escalated", halted=outcome.halted, why=outcome.why,
                          packet=len(escalation(run, outcome).splitlines()))
        with self._lock:
            self.waiting[run_id] = list(outcome.held)
        return Reply(run_id, outcome.answer, outcome.delivered, outcome.halted,
                     run.calls, tuple(outcome.held))

    def deadline(self, run_id: str, at: float) -> "Deadline":
        return Deadline(at, self.now,
                        lambda name: self.log.emit(run_id, "deadline", worker=name))

    def decide(self, run_id: str, reviewer: str, approve: bool, why: str = "") -> list:
        """A person's decision on what a run held, applied by your code. This is
        Chapter 11's review, with the one thing a deployment adds: the decision
        is written down next to the run it belongs to."""
        with self._lock:
            actions = self.waiting.pop(run_id, [])
        lines = []
        for action in actions:
            lines.append(review(action, reviewer, self.ledger, approve, why))
            self.log.emit(run_id, "reviewed", reviewer=reviewer, approved=approve,
                          booking=action.booking, amount_usd=action.amount_usd)
        return lines

    def resolve(self, run_id: str, reviewer: str, note: str) -> None:
        """An escalated run, answered by a person. Without this event the log
        can say how often the system asks for help and never how often it
        gets any."""
        self.log.emit(run_id, "resolved", reviewer=reviewer, note=note)


def counting_ids(prefix: str = "r") -> Callable:
    """Run ids that are reproducible, for a demonstration and for tests. A
    deployment uses the default instead: a counter restarts at one when the
    process does, and two processes behind the same address would hand the
    same id to two different travellers."""
    count, lock = [0], threading.Lock()

    def next_id() -> str:
        with lock:
            count[0] += 1
            return f"{prefix}-{count[0]:04d}"
    return next_id


# --- Emit: one line per decision, in a form a program can read ----------------

class EventLog:
    """Newline-delimited JSON, appended under a lock.

    Chapter 11's record is prose, written for a person who was not there, and
    it stays exactly as it was: every `answered` event carries it verbatim.
    This is the same decisions as fields, for the program that has to answer
    questions about ten thousand runs rather than read one.
    """

    def __init__(self, path: str, release: Release, now: Callable = time.time):
        self.path, self.release, self.now = path, release, now
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def emit(self, run: str, event: str, **fields) -> dict:
        record = {"at": round(self.now(), 3), "run": run, "event": event,
                  "service": self.release.service, "version": self.release.version,
                  "model": self.release.model, **fields}
        line = json.dumps(record)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as out:
                out.write(line + "\n")
                out.flush()
        return record

    def read(self) -> list:
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as lines:
            return [json.loads(line) for line in lines if line.strip()]


# --- The front door -----------------------------------------------------------

def handler_for(service: Service):
    """POST /ask answers a question; GET /health says whether the process is
    up. The health check calls no model: a probe that spends money turns a
    load balancer into a customer."""

    class Ask(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            if self.path != "/ask":
                return self.send_error(404)
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                sent = json.loads(body)
                request = Request(sent["question"], sent["traveller"], sent["day"])
            except (ValueError, KeyError, TypeError):
                return self.send_error(400, "expected JSON: question, traveller, day")
            reply = service.answer(request)
            self.reply(200, {"run": reply.run, "answer": reply.answer,
                             "delivered": reply.delivered, "halted": reply.halted})

        def do_GET(self):
            if self.path != "/health":
                return self.send_error(404)
            self.reply(200, {"ok": True, "version": service.log.release.version})

        def reply(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass                       # the event log is the log

    return Ask


def serve(service: Service, port: int = 8080) -> ThreadingHTTPServer:
    """A thread per request, which is why nothing in Service holds per-run state
    on the instance."""
    return ThreadingHTTPServer(("127.0.0.1", port), handler_for(service))


# --- Demonstration: a day of traffic, with scripted stand-ins -----------------

import shutil                                                          # noqa: E402
import urllib.request                                                  # noqa: E402

from combined import (BAG_FACT, BAG_LESSON, CREDIT_ANSWER, CREDIT_Q,   # noqa: E402
                      CREDIT_TASK, EARLY_LESSON, POLICY_REFUND, QUOTED_DRAFT,
                      QUOTED_FACT, QUOTED_FIXED, QUOTED_Q, REFUND_TASK,
                      SERVICE_REPLIES)
from guardrails import (CANCEL_ANSWER, CANCEL_Q, FARE_TASK,            # noqa: E402
                        FARES_CANCEL, credit, recorded)
from orchestration import (BAG_FEE, CANCEL, MERGED, PRICE_B, Scripted,  # noqa: E402
                           delegate, lookups, scripted_runner)
from policy_retrieval import QUESTION                                   # noqa: E402
from self_critique import ScriptedModel                                 # noqa: E402

FOLDER, DAY_ONE, DAY_TWO = "deployed", "2026-06-02", "2026-06-09"
START = 1780000000.0                     # a fixed instant, so the log is reproducible

PRICING = {"fares": [lookups("B") + [BAG_FEE],
                     [("calculate", {"expression": "175 + 40"})],
                     "Flight B costs $175 and takes 3.5 hours. It is longer than 3 "
                     "hours, so the first checked bag costs $40 [baggage-2026#1], "
                     "which makes the total $215."],
           "policy": [[("search_policies", {"query": "cancel within 24 hours refund"})],
                      "Cancelling 20 hours before departure is within 24 hours: the "
                      "fare is not refunded but becomes travel credit, minus a $50 "
                      "cancellation fee [refunds#2]."]}
COMPUTE = "The fare is $175 and a $50 cancellation fee applies. Compute the travel credit."
CANCEL_PLAN = [[delegate("fares", FARE_TASK), delegate("policy", REFUND_TASK)],
               [delegate("fares", COMPUTE)],
               [delegate("service", "Request $125 travel credit on BK-4471: the $175 "
                                    "fare minus the $50 cancellation fee [refunds#2].")],
               CANCEL_ANSWER]
CANCEL_TEAM = {"fares": FARES_CANCEL, "policy": POLICY_REFUND,
               "service": [[credit("BK-4471", 125, "fare $175 minus the $50 "
                                                   "cancellation fee [refunds#2]")],
                           recorded("BK-4471", 125)]}
# The same plan, answered without the step the deadline refuses: the fare and the
# rule, and no credit, because nothing computed one.
PARTIAL = [CANCEL_PLAN[0], CANCEL_PLAN[1], CANCEL_ANSWER.split(" The credit comes to")[0]]

# (traveller, question) -> what each stand-in says, and what the run may write.
SCRIPT = {
    ("T-88", QUESTION): ([[delegate("fares", PRICE_B), delegate("policy", CANCEL)],
                          MERGED], PRICING, [("critique", [])],
                         (BAG_FACT,), (EARLY_LESSON,)),
    ("T-41", QUOTED_Q): ([[delegate("policy", REFUND_TASK)], QUOTED_DRAFT],
                         {"policy": POLICY_REFUND},
                         [("revise", QUOTED_FIXED), ("critique", [])],
                         (QUOTED_FACT,), (BAG_LESSON,)),
    ("T-41", CREDIT_Q): ([[delegate("policy", REFUND_TASK)],
                          [delegate("service", CREDIT_TASK)], CREDIT_ANSWER],
                         {"policy": POLICY_REFUND, "service": SERVICE_REPLIES},
                         [("critique", [])], (), ()),
    ("T-63", CANCEL_Q): (CANCEL_PLAN, CANCEL_TEAM, [("critique", [])], (), ()),
    ("T-52", CANCEL_Q): (PARTIAL, CANCEL_TEAM, [("critique", [])], (), ()),
}


def scripted_engine(script: dict) -> Callable:
    """The engine the service calls. In a deployment this builds real agents; here
    it looks its replies up by who asked and what they asked, so that the same
    traffic produces the same log on every machine. Nothing it prints measures a
    model: the stand-ins were written to produce one good day and two bad
    moments, and that is all their numbers show."""
    def engine(request: Request, stores: Stores, ledger: Ledger, limits: Limits,
               composition: Composition, deadline: Deadline) -> tuple:
        coordinator, workers, critic, facts, notes = script[(request.traveller,
                                                             request.question)]
        return handle(request.question, request.traveller, request.day,
                      Scripted(*coordinator), ScriptedModel(critic),
                      lambda built: deadline.guard(scripted_runner(
                          {name: Scripted(*said) for name, said in workers.items()})),
                      ledger, stores, facts, notes, limits=limits,
                      composition=composition)
    return engine


class StepClock:
    """Stands in for the clock: every reading is the previous one plus a fixed
    step, so elapsed times in the log are the same on every machine."""

    def __init__(self, start: float = START, step: float = 0.5):
        self.at, self.step, self._lock = start, step, threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            self.at += self.step
            return self.at

    def skip(self, minutes: float) -> None:
        self.at += minutes * 60


TRAFFIC = [("A question about a fare", "T-88", DAY_ONE, QUESTION),
           ("A traveller repeats what support told them", "T-41", DAY_ONE, QUOTED_Q),
           ("A week later, that number comes back", "T-41", DAY_TWO, CREDIT_Q),
           ("A cancellation, with money to move", "T-63", DAY_TWO, CANCEL_Q),
           ("The same request, after the deadline is tightened",
            "T-52", DAY_TWO, CANCEL_Q)]


def ask(port: int, traveller: str, day: str, question: str) -> dict:
    """What a caller does. Nothing else in this file is the caller's business."""
    body = json.dumps({"question": question, "traveller": traveller,
                       "day": day}).encode()
    call = urllib.request.Request(f"http://127.0.0.1:{port}/ask", body,
                                  {"Content-Type": "application/json"})
    with urllib.request.urlopen(call) as answered:
        return json.loads(answered.read())


def run_demo() -> None:
    shutil.rmtree(FOLDER, ignore_errors=True)       # the demonstration starts clean
    clock, ledger = StepClock(), Ledger()
    log = EventLog(f"{FOLDER}/events.jsonl", Release(), clock)
    service = Service(FOLDER, log, ledger, scripted_engine(SCRIPT),
                      ids=counting_ids(), now=clock)
    server = serve(service, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as up:
        print(f"GET /health  {json.loads(up.read())}\n")

    replies = []
    for title, traveller, day, question in TRAFFIC:
        if traveller == "T-52":
            service.deadline_s = 2.0                # the operator tightens it
            print("--- the deadline is cut to 2 seconds ---\n")
        print(f"=== {title} ===")
        reply = ask(port, traveller, day, question)
        replies.append(reply)
        print(f"  POST /ask  {traveller}  {day}  ->  {reply['run']}  "
              + ("delivered" if reply["delivered"] else f"halted: {reply['halted']}"))
        print(f"  {reply['answer']}\n")

    print("=== What a person did next ===")
    clock.skip(11)                                  # eleven minutes later
    print("  " + "\n  ".join(service.decide(replies[3]["run"], "dana", approve=True)))
    clock.skip(37)                                  # and thirty-seven more
    print("  " + "\n  ".join(service.decide(replies[2]["run"], "raj", approve=False,
                                            why="no record of that quote")))
    service.resolve(replies[2]["run"], "raj", "told the traveller what the policy gives")
    print()

    print("=== Three of the lines the service wrote ===")
    wanted = {("r-0003", "held"), ("r-0003", "escalated"), ("r-0005", "deadline")}
    for written in log.read():
        if (written["run"], written["event"]) in wanted:
            print("  " + json.dumps(written))
    server.shutdown()
    server.server_close()


if __name__ == "__main__":
    run_demo()
