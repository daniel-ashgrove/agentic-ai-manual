"""
Evaluation and feedback, standard library only.

The agent is run on a fixed set of cases, several times each, and every answer
is graded against a reference a person wrote. What a run's scores may change in
the next run is decided here too. Three decisions stay with your code, not the
model:

    grade     what counts as right, and who wrote the reference
    compare   against what: the same cases, several trials, a stated cost
    adopt     what a score may change in future runs
"""

import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from policy_retrieval import (DOCUMENTS, QUESTION, PolicyLibrary,     # Chapter 6
                              check_answer, make_search_contract)
# Chapter 8, for the demonstration and the attempt records
from self_critique import (GROUNDED, LESSONS, SESSIONS, Finding, LessonStore,
                           ScriptedModel, check_findings, gather_evidence,
                           make_lesson_contract, record_attempts, refine,
                           review_report, run_session)
from tool_contracts import CONTRACTS, FLIGHTS, ToolRegistry           # Chapter 5


# --- Cases, references, and the reference grader ----------------------------

def bag_total(flight_id: str, bags: int = 1) -> int:
    """baggage-2026#1 as a person read it, written once as code: the first bag is
    free on flights of 3 hours or less and $40 on longer ones; each other bag $60."""
    flight = FLIGHTS[flight_id]
    first = 0 if flight["duration_hours"] <= 3 else 40
    return flight["price_usd"] + (first + 60 * (bags - 1) if bags else 0)


def cheapest_arriving_before(time: str, bags: int = 1) -> str:
    in_time = [f for f in FLIGHTS if FLIGHTS[f]["arrival"] < time]
    return min(in_time, key=lambda f: bag_total(f, bags))


@dataclass(frozen=True)
class Case:
    case_id: str
    question: str
    must_state: tuple            # what a right answer states: amounts or short phrases
    must_not_state: tuple = ()   # wrong claims already seen, kept as regression tests
    reference: str = ""          # the right answer in a person's words, for reading


BEST = cheapest_arriving_before("18:00")

CASES = (
    Case("bag-and-cancel", QUESTION,
         (f"${bag_total('B')}", "travel credit", "$50"), ("refunded in full",),
         "Flight B with one bag is $215. Cancelling 20 hours out is within 24 hours: "
         "travel credit, minus a $50 fee."),
    Case("cheapest-with-bag",
         "Which flight gets me there before 18:00 for the least money? "
         "I'm checking one bag.",
         (f"flight {BEST}", f"${bag_total(BEST)}"), (),
         "Flight A: its first bag is free, so it costs $210 against flight B's $215."),
    Case("bag-A",
         "I'm booking flight A and checking one bag. What will I pay in total?",
         (f"${bag_total('A')}",), ("$250",),   # $250: the long-flight fee, misapplied
         "$210: flight A takes 2.5 hours, so the first checked bag is free."),
    Case("cancel-early",
         "If I cancel flight C three days before departure, what happens to my fare?",
         ("refunded in full",), ("travel credit",),
         "Three days is more than 24 hours, so the fare is refunded in full."),
)


def _states(answer: str, phrase: str) -> bool:
    """The phrase as a whole: '$50' is not in '$500', nor 'flight a' in
    'flight arrives'."""
    return re.search(r"(?<![\w$])" + re.escape(phrase.lower()) + r"(?!\w)",
                     answer.lower()) is not None


def grade(case: Case, answer: Optional[str]) -> list[str]:
    """The reference grader. Empty means correct. It reads for facts a person decided
    a correct answer must contain, which the check and the critic never had."""
    if answer is None:
        return ["withheld"]
    return ([f"does not state {p!r}" for p in case.must_state if not _states(answer, p)]
            + [f"states {p!r}" for p in case.must_not_state if _states(answer, p)])


@dataclass(frozen=True)
class Trial:
    """What one run of the system under test produced: the harness grades this,
    not the path."""
    case_id: str
    trial: int
    answer: Optional[str]            # None: the system withheld its answer
    tool_results: tuple
    model_calls: int
    tokens: int = 0                  # when the system can count them; 0 when it can't


@dataclass(frozen=True)
class Score:
    case_id: str
    trial: int
    outcome: str                     # "correct", "wrong" or "withheld"
    traceable: bool                  # Chapter 6's check, on what the system delivered
    problems: tuple
    model_calls: int
    tokens: int = 0


def score_trial(case: Case, t: Trial) -> Score:
    problems = grade(case, t.answer)
    outcome = "withheld" if t.answer is None else "wrong" if problems else "correct"
    traceable = t.answer is not None and not check_answer(t.answer, case.question,
                                                          list(t.tool_results))
    return Score(case.case_id, t.trial, outcome, traceable, tuple(problems),
                 t.model_calls, t.tokens)


# --- Runs, comparisons, the score log, and decisions ------------------------

System = Callable[[Case, int], Trial]


def run_eval(system: System, cases: tuple = CASES, trials: int = 3) -> list[Score]:
    """Every case, every trial, in the same order for every system, so runs pair up."""
    return [score_trial(case, system(case, n))
            for case in cases for n in range(1, trials + 1)]


def summarize(scores: list[Score]) -> dict:
    outcomes = [s.outcome for s in scores]
    by_case: dict[str, list[bool]] = {}
    for s in scores:
        by_case.setdefault(s.case_id, []).append(s.outcome == "correct")
    delivered = [s for s in scores if s.outcome != "withheld"]
    return {"trials": len(scores), "correct": outcomes.count("correct"),
            "wrong": outcomes.count("wrong"), "withheld": outcomes.count("withheld"),
            "traceable": sum(s.traceable for s in delivered),
           "delivered": len(delivered),
            "model_calls": sum(s.model_calls for s in scores),
            "tokens": sum(s.tokens for s in scores), "cases": len(by_case),
            "every_trial_correct": sum(all(v) for v in by_case.values()),    # pass^k
            "some_trial_correct": sum(any(v) for v in by_case.values())}     # pass@k


@dataclass(frozen=True)
class Comparison:
    fixed: int                  # trials the second run got right and the first did not
    broke: int                  # trials the first run got right and the second did not
    better_cases: int
    worse_cases: int
    mean_difference: float      # second minus first, per case, trials averaged
    standard_error: float
    extra_calls: int


def compare(first: list[Score], second: list[Score]) -> Comparison:
    """A paired comparison: the same cases and trials, matched one to one."""
    pairs = {(s.case_id, s.trial): [s] for s in first}
    for s in second:
        pairs.setdefault((s.case_id, s.trial), []).append(s)
    if any(len(pair) != 2 for pair in pairs.values()):
        raise ValueError("the two runs do not cover the same cases and trials")
    ok = {key: [s.outcome == "correct" for s in pair] for key, pair in pairs.items()}
    per_case: dict[str, list[int]] = {}
    for (case_id, _), (a, b) in ok.items():
        per_case.setdefault(case_id, []).append(int(b) - int(a))
    diffs = [sum(d) / len(d) for d in per_case.values()]
    se = statistics.stdev(diffs) / len(diffs) ** 0.5 if len(diffs) > 1 else float("nan")
    return Comparison(sum(b and not a for a, b in ok.values()),
                      sum(a and not b for a, b in ok.values()),
                      sum(d > 0 for d in diffs), sum(d < 0 for d in diffs),
                      statistics.fmean(diffs), se,
                      sum(s.model_calls for s in second)
                      - sum(s.model_calls for s in first))


class ScoreLog:
    """One JSON line per run, appended and never rewritten. A later run is judged
    against what this file says earlier runs scored, under the configuration they
    ran."""

    def __init__(self, path: str):
        self.path = Path(path)

    def record(self, run_id: str, config: dict, scores: list[Score], **extra) -> dict:
        entry = {"run": run_id, "config": config, **summarize(scores), **extra,
                 "scores": [asdict(s) for s in scores]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def runs(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]


def decide_lesson(with_it: list[Score], without_it: list[Score]) -> tuple[str, str]:
    """Adopt, second half: a lesson stays only while the agent does no worse with it.
    The comparison runs on every case, not only the one the lesson was written from."""
    c = compare(with_it, without_it)
    if c.fixed and c.broke:
        return "review", (f"removing it fixed {c.fixed} trials and broke {c.broke}; "
                          f"a person decides")
    if c.fixed:
        return "retire", f"removing it fixed {c.fixed} trials and broke none"
    if c.broke:
        return "keep", f"removing it broke {c.broke} trials"
    return "keep", f"removing it changed no grade; model calls {c.extra_calls:+d}"


_ATTEMPT = re.compile(r"attempt-\d+ (passed|failed)")


def attempt_record(store: LessonStore, run_id: str) -> dict:
    """Chapter 8's episodes for one run: how much work each answer took, by the
    agent's own account. Nothing here says whether an answer was right."""
    runs = [_ATTEMPT.findall(e.text) for e in store.episodes()
            if e.session.startswith(run_id + "/")]
    return {"revisions_per_answer": sum(len(r) - 1 for r in runs) / len(runs),
            "first_drafts_passing": sum(r[0] == "passed" for r in runs)}


# --- Demonstration: scripted systems, one lesson store, a score log ----------

FOLDER, AGENT, LOG = "evals", "flight-agent", "evals/runs.jsonl"


@dataclass(frozen=True)
class Script:
    calls: tuple         # the tool calls this trial makes before its first draft
    replies: tuple       # (step, reply) pairs, in the order Chapter 8's loop asks for them


def _lookups(*flights):
    return tuple(("look_up_flight", {"flight_id": f}) for f in flights)


BAG_FEE = (("search_policies", {"query": "checked bag fee"}),)
FARE_B = _lookups("B") + BAG_FEE + (("search_policies", {"query": "cancel within 24 hours "
                                                                 "refund"}),
                                    ("calculate", {"expression": "175 + 40"}))
ALL_FLIGHTS = _lookups("A", "B", "C") + BAG_FEE + (("calculate", {"expression": "175 + 40"}),)
EARLY_CANCEL = (("search_policies", {"query": "cancel more than 24 hours refund"}),)

UNCITED, MISGROUNDED = SESSIONS["2026-09-21"][0][1], SESSIONS["2026-09-22"][0][1]
SPLIT = ("Flight B is the cheapest flight that arrives before 18:00: it costs $175 and arrives "
         "at 16:45. It takes 3.5 hours, so the first checked bag costs $40 [baggage-2026#1], "
         "which makes the total $215.")
CHEAPEST = ("Flight A is the cheapest flight that gets you there before 18:00 with your bag: "
            "it arrives at 17:30 and takes 2.5 hours, so the first checked bag is free "
            "[baggage-2026#1], and you will pay $210. Flight B has the lower fare, $175, but "
            "it takes 3.5 hours, so its first checked bag costs $40 [baggage-2026#1] and the "
            "total is $215.")
BAG_A = ("Flight A costs $210 and takes 2.5 hours. On flights of 3 hours or less the first "
         "checked bag is free [baggage-2026#1], so you will pay $210.")
BAG_A_WITH_FEE = ("Flight A costs $210 and takes 2.5 hours. The first checked bag costs $40 "
                  "[baggage-2026#1], which makes the total $250.")
EARLY = ("Three days before departure is more than 24 hours before, so the fare is refunded in "
         "full to the original form of payment [refunds#1].")


def _passes(draft):
    return (("draft", draft), ("critique", []))


PLAIN = {                # the first prompt Chapter 8's agent used
    "bag-and-cancel": [Script(FARE_B, tuple(s)) for s in SESSIONS.values()],  # Chapter 8's
    "cheapest-with-bag": [Script(ALL_FLIGHTS, (
        ("draft", SPLIT),
        ("critique", ["Flight A takes 2.5 hours, so its first bag is free [baggage-2026#1]: "
                      "with the bag it costs $210, less than flight B's $215."]),
        ("revise", CHEAPEST), ("critique", [])))],
    "bag-A": [Script(_lookups("A") + BAG_FEE, _passes(BAG_A))],
    "cancel-early": [Script(EARLY_CANCEL, _passes(EARLY))],
}
STATED = dict(PLAIN, **{  # a first prompt stating every requirement a person could write down
    "bag-and-cancel": [Script(FARE_B, (("draft", d),))
                       for d in (GROUNDED, MISGROUNDED, GROUNDED)],
    "cheapest-with-bag": [Script(ALL_FLIGHTS, (("draft", CHEAPEST),))],
})
WITH_LESSON = {          # (lesson shown, case, trial; None: every trial) -> the changed draft
    ("lesson-1", "bag-and-cancel", 1): Script(FARE_B, _passes(GROUNDED)),
    ("lesson-2", "bag-A", None): Script(_lookups("A") + BAG_FEE + (
        ("calculate", {"expression": "210 + 40"}),), _passes(BAG_A_WITH_FEE)),
}


def scripted_system(drafts: dict, max_revisions: int, store: Optional[LessonStore] = None,
                    run_id: str = "", shown: tuple = ()) -> System:
    """Chapter 8's loop, with a stand-in whose replies are written by hand. A trial is
    recorded as an episode in the store when one is given, as Chapter 8's agent does."""
    def system(case: Case, n: int) -> Trial:
        script = drafts[case.case_id][(n - 1) % len(drafts[case.case_id])]
        for lesson in shown:
            script = WITH_LESSON.get((lesson, case.case_id, n),
                                     WITH_LESSON.get((lesson, case.case_id, None), script))
        registry = ToolRegistry(CONTRACTS + [make_search_contract(PolicyLibrary(DOCUMENTS))])
        results = [registry.execute(name, args).content for name, args in script.calls]
        model = ScriptedModel(list(script.replies))
        result = refine(model("draft"),
                        check=lambda a: check_findings(a, case.question, results),
                        critique=lambda a: [Finding("critic", "critic", t)
                                            for t in model("critique", a)],
                        revise=lambda a, f: model("revise", review_report(f)),
                        max_revisions=max_revisions)
        if store is not None:
            record_attempts(store, result, f"{run_id}/{case.case_id}/{n}")
        answer = result.delivered.answer if result.delivered else None
        return Trial(case.case_id, n, answer, tuple(results), result.model_calls)
    return system


def keep_chapter_8_lessons(store: LessonStore) -> None:
    """Chapter 8's first session, replayed through its lesson gate: what it kept."""
    result = run_session(SESSIONS["2026-09-21"], gather_evidence())
    episode = record_attempts(store, result, "2026-09-21")
    registry = ToolRegistry([make_lesson_contract(store, result, "2026-09-21",
                                                  episode.memory_id)])
    for args in LESSONS["2026-09-21"]:
        registry.execute("keep_lesson", args)


def show_scores(scores: list[Score]) -> None:
    for case in CASES:
        mine = [s for s in scores if s.case_id == case.case_id]
        outcomes = ", ".join(s.outcome if s.outcome == "correct" else s.outcome.upper()
                             for s in mine)
        calls = ", ".join(str(s.model_calls) for s in mine)
        print(f"  {case.case_id:18} {outcomes:28} {calls} model calls")
        for s in mine:
            if s.outcome == "wrong":
                print(f"      trial {s.trial}: {'; '.join(s.problems)}")
    t = summarize(scores)
    print(f"  Traceable: {t['traceable']} of {t['delivered']} delivered answers")
    print(f"  Correct:   {t['correct']} of {t['trials']} trials; every trial correct on "
          f"{t['every_trial_correct']} of {t['cases']} cases, at least one on "
          f"{t['some_trial_correct']}")


def show_comparison(label: str, c: Comparison) -> None:
    print(f"  {label}: fixed {c.fixed}, broke {c.broke}; "
          f"{c.extra_calls:+d} model calls")
    print(f"      per case: better on {c.better_cases} of {len(CASES)}, worse on "
          f"{c.worse_cases}; mean {c.mean_difference:+.2f}, standard error "
          f"{c.standard_error:.2f}")


def run_demo() -> None:
    store, log = LessonStore(FOLDER, AGENT), ScoreLog(LOG)
    store.forget_all()                                 # the demonstration starts clean
    Path(LOG).unlink(missing_ok=True)

    def run(run_id: str, config: dict, system: System) -> list[Score]:
        scores = run_eval(system)
        extra = attempt_record(store, run_id) if config["system"] == "critique" else {}
        log.record(run_id, config, scores, **extra)
        return scores

    print(f"Eval set: {len(CASES)} cases, references written by a person; 3 trials each\n")
    print("=== 1. Chapter 8's critique loop, graded two ways ===")
    a = run("A", {"system": "critique", "prompt": "plain", "lessons": []},
            scripted_system(PLAIN, 2, store, "A"))
    show_scores(a)

    print("\n=== 2. Three systems on the same cases and trials ===")
    single = run("single", {"system": "one call", "prompt": "plain", "lessons": []},
                 scripted_system(PLAIN, 0))
    stated = run("stated", {"system": "one call", "prompt": "stated", "lessons": []},
                 scripted_system(STATED, 0))
    print(f"  {'system':26} correct  wrong  withheld  model calls")
    for name, scores in (("one call", single), ("one call, stated prompt", stated),
                         ("critique loop", a)):
        t = summarize(scores)
        print(f"  {name:26} {t['correct']:>7}  {t['wrong']:>5}  {t['withheld']:>8}  "
              f"{t['model_calls']:>11}")
    show_comparison("critique loop vs one call", compare(single, a))
    show_comparison("critique loop vs stated prompt", compare(stated, a))

    print("\n=== 3. Lessons, judged on every case ===")
    keep_chapter_8_lessons(store)
    shown = tuple(l.memory_id for l in store.lessons())
    print("  Kept by Chapter 8's first session: " + ", ".join(
        f"{l.memory_id} ({l.key})" for l in store.lessons()))
    b = run("B", {"system": "critique", "prompt": "plain", "lessons": list(shown)},
            scripted_system(PLAIN, 2, store, "B", shown))
    t = summarize(b)
    print(f"  with both:          correct {t['correct']} of {t['trials']}, "
          f"{t['model_calls']} model calls")
    decisions = {}
    for lesson in shown:
        rest = tuple(x for x in shown if x != lesson)
        without = run(f"B-without-{lesson}", {"system": "critique", "prompt": "plain",
                                              "lessons": list(rest)},
                      scripted_system(PLAIN, 2, store, f"B-without-{lesson}", rest))
        decisions[lesson] = decide_lesson(b, without)
        t = summarize(without)
        print(f"  without {lesson}:   correct {t['correct']} of {t['trials']}, "
              f"{t['model_calls']} model calls")
        print(f"      -> {decisions[lesson][0]} {lesson}: {decisions[lesson][1]}")
    for lesson, (decision, _) in decisions.items():
        if decision == "retire":
            store.forget(lesson)
    now = tuple(l.memory_id for l in LessonStore(FOLDER, AGENT).lessons())
    run("C", {"system": "critique", "prompt": "plain", "lessons": list(now)},
        scripted_system(PLAIN, 2, store, "C", now))
    print(f"  The next run is shown: {', '.join(now) or 'no lessons'}")

    print("\n=== 4. The score log: Chapter 8's agent, run by run ===")
    print(f"  {'run':4} {'lessons shown':19} {'revisions/answer':>16}  "
          f"{'first drafts passing':>20}  {'correct':>8}  {'calls':>5}")
    for e in log.runs():
        if e["run"] in ("A", "B", "C"):
            print(f"  {e['run']:4} {', '.join(e['config']['lessons']) or 'none':19} "
                  f"{e['revisions_per_answer']:>16.2f}  "
                  f"{str(e['first_drafts_passing']) + ' of ' + str(e['trials']):>20}  "
                  f"{str(e['correct']) + ' of ' + str(e['trials']):>8}  {e['model_calls']:>5}")


if __name__ == "__main__":
    run_demo()
