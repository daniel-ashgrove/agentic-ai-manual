"""
Critique and revision, standard library only.

An answer is checked, critiqued and revised before anyone sees it, and what the
agent learns from its own failed drafts is kept only through a gate. Three
decisions stay with your code, not the model:

    stop      when an answer is finished: the checks decide; the critic can object
    budget    how many revisions one answer may cost
    learn     which lessons may be kept, and on what evidence
"""

import re
from dataclasses import dataclass, replace
from typing import Callable, Optional

from agent_memory import Memory, MemoryStore                         # Chapter 7
from policy_retrieval import (DOCUMENTS, QUESTION, PolicyLibrary,    # Chapter 6
                              check_answer, make_search_contract, tokenize)
from tool_contracts import CONTRACTS, ToolContract, ToolRegistry    # Chapter 5


# --- Findings, the loop, and the review the model is shown ------------------

@dataclass(frozen=True)
class Finding:
    source: str          # "check": your code found it. "critic": a model thinks so.
    rule: str            # the check rule that fired, or "critic"
    detail: str


CHECK_RULES = {          # Chapter 6's check, by words each of its problems contains
    "unretrieved-citation": "which no search returned",
    "not-current": "which is not current policy",
    "unsourced-number": "with no source",
    "uncited-claim": "makes a claim with no citation",
}


def _rule_of(problem: str) -> str:
    for rule, words in CHECK_RULES.items():
        if words in problem:
            return rule
    raise ValueError(f"no rule matches the check's problem {problem!r}")


def check_findings(answer: str, question: str,
                   tool_results: list[str]) -> list[Finding]:
    """Chapter 6's check, with each problem labeled by the rule that fired."""
    return [Finding("check", _rule_of(p), p)
            for p in check_answer(answer, question, tool_results)]


@dataclass(frozen=True)
class Attempt:
    number: int
    answer: str
    checks: tuple = ()           # what the checks found; empty means it passes
    critique: tuple = ()         # the critic's objections, if the critic was asked

    @property
    def passes(self) -> bool:
        return not self.checks


@dataclass(frozen=True)
class Refinement:
    attempts: tuple
    delivered: Optional[Attempt]     # None: nothing passed the checks, so withhold
    model_calls: int
    stopped: str


def refine(draft: str, check: Callable, critique: Callable, revise: Callable,
           max_revisions: int = 2) -> Refinement:
    """Check, critique and revise one answer until it passes the checks and the
    critic has no objection, or the revision budget is spent.

        check(answer)             -> [Finding]   your code, no model call
        critique(answer)          -> [Finding]   one model call
        revise(answer, findings)  -> str         one model call
    """
    attempts, calls = [Attempt(1, draft, tuple(check(draft)))], 1
    while True:
        current = attempts[-1]
        budget_left = len(attempts) - 1 < max_revisions
        if current.passes:
            if not budget_left:          # an objection nobody can act on costs a call
                return Refinement(tuple(attempts), current, calls,
                                  "passed the checks; no revision left to spend")
            objections = tuple(critique(current.answer))
            calls += 1
            attempts[-1] = current = replace(current, critique=objections)
            if not objections:
                return Refinement(tuple(attempts), current, calls,
                                  "passed the checks; the critic had no objection")
        elif not budget_left:
            break
        new = revise(current.answer, list(current.checks or current.critique))
        calls += 1
        attempts.append(Attempt(len(attempts) + 1, new, tuple(check(new))))
    passing = [a for a in attempts if a.passes]
    return Refinement(tuple(attempts), passing[-1] if passing else None, calls,
                      "revision budget spent" + ("" if passing else "; withheld"))


REVIEW_HEADER = ("Review of your previous answer, from the application. Check findings "
                 "are facts about the answer. Reviewer findings are a model's opinion: "
                 "weigh them against the tool results before you change anything.")


def review_report(findings: list[Finding]) -> str:
    """What the model is shown when it is asked to revise, labeled by who found what."""
    lines = [REVIEW_HEADER, ""]
    for f in findings:
        who = f"check: {f.rule}" if f.source == "check" else "reviewer"
        lines.append(f"- ({who}) {' '.join(f.detail.split())}")
    return "\n".join(lines)


# --- Remembering attempts, and the gate for lessons -------------------------

class LessonStore(MemoryStore):
    """Chapter 7's store, opened for the agent instead of a traveller. Episodes
    record how each answer went; lessons record what a failed draft taught.
    A lesson's quote field holds its evidence: a fact quotes the traveller, a
    lesson quotes your checks."""

    def _next_id(self, kind: str) -> str:
        self.counters.setdefault(kind, 0)
        return super()._next_id(kind)

    def lessons(self) -> list[Memory]:
        return [r for r in self.records
                if r.kind == "lesson" and r.superseded_by is None]

    def add_lesson(self, rule: str, text: str, evidence: str, session: str) -> Memory:
        """One current lesson per check rule; the one it replaces is kept for audit."""
        new = Memory(self._next_id("lesson"), "lesson", session, text, rule, evidence)
        self.records = [replace(r, superseded_by=new.memory_id)
                        if r in self.lessons() and r.key == rule else r
                        for r in self.records] + [new]
        self._save()
        return new


def _outcome(a: Attempt) -> str:
    result = ("passed" if a.passes
              else "failed " + ", ".join(sorted({f.rule for f in a.checks})))
    return f"attempt-{a.number} {result}" + (", critic objected" if a.critique else "")


def record_attempts(store: LessonStore, result: Refinement, session: str) -> Memory:
    """Your code writes what happened: rule names and counts, nobody's words."""
    end = (f"delivered attempt-{result.delivered.number}" if result.delivered
           else "withheld")
    steps = "; ".join(_outcome(a) for a in result.attempts)
    return store.add_episode(f"{steps}; {end}; {result.model_calls} model calls",
                             session)


def attempt_history(result: Refinement) -> str:
    """What the model reads before it proposes a lesson: each attempt and its
    findings."""
    lines = []
    for a in result.attempts:
        lines.append(f"Attempt {a.number}: {a.answer}")
        lines += [f"  - ({f.source}: {f.rule}) {f.detail}"
                  for f in a.checks + a.critique]
        if a.passes and not a.critique:
            lines.append("  - passed every check")
    return "\n".join(lines)


_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_RULE_WORDS = re.compile(r"\b(always|never|must|should|every|do not|don't)\b")


def _words(text: str) -> set[str]:
    return {w for w in tokenize(text) if not w.replace(".", "").isdigit()}


def check_lesson(result: Refinement, args: dict) -> list[str]:
    """Gate 2 for keep_lesson: a check recorded the failure this session, and a later
    attempt passed every check. Traceable, not true: a real failure can still be
    turned into the wrong lesson."""
    attempts = {a.number: a for a in result.attempts}
    failed, fixed = attempts.get(args["failed"]), attempts.get(args["fixed"])
    evidence = " ".join(f.detail for f in (failed.checks if failed else ())
                        if f.rule == args["rule"])
    note, problems = args["note"], []
    if not evidence:
        problems.append(f"attempt {args['failed']} did not fail the {args['rule']!r} "
                        f"check in this session")
    elif fixed is None or fixed.number <= failed.number or not fixed.passes:
        problems.append(f"attempt {args['fixed']} is not a later attempt that passed "
                        f"every check")
    if _RULE_WORDS.search(note.lower().replace("\u2019", "'")):
        problems.append("the note states a rule; describe what went wrong and what "
                        "the passing revision changed, without words like 'always' "
                        "or 'every'")
    if evidence and not _words(note) & _words(evidence):
        problems.append("the note shares no words with what the check found")
    missing = sorted(set(_NUMBER.findall(note)) - set(_NUMBER.findall(evidence)))
    if evidence and missing:
        problems.append(f"the note states {', '.join(missing)}, which the check's "
                        f"finding does not")
    return problems


def make_lesson_contract(store: LessonStore, result: Refinement, session: str,
                         episode_id: str) -> ToolContract:
    def keep_lesson(rule: str, failed: int, fixed: int, note: str) -> str:
        evidence = (f"{episode_id}: attempt-{failed} failed {rule}; "
                    f"attempt-{fixed} passed")
        new = store.add_lesson(rule, " ".join(note.split()), evidence, session)
        return f"Kept as [{new.memory_id}]."

    return ToolContract(
        name="keep_lesson",
        description=("Keep a lesson from this session's attempts for future drafts: "
                     "the check rule that failed, the attempt that failed it, the "
                     "later attempt that passed every check, and one sentence on what "
                     "the revision changed. Only a check's finding is evidence; a "
                     "reviewer's objection is not. Describe what happened; do not "
                     "state a rule."),
        input_schema={
            "type": "object",
            "properties": {
                "rule": {"type": "string", "enum": sorted(CHECK_RULES),
                         "description": "The check rule that failed."},
                "failed": {"type": "integer", "minimum": 1,
                           "description": "The attempt that failed it."},
                "fixed": {"type": "integer", "minimum": 1,
                          "description": "The later attempt that passed every check."},
                "note": {"type": "string", "minLength": 1, "maxLength": 160,
                         "description": "What the passing revision changed."},
            },
            "required": ["rule", "failed", "fixed", "note"],
            "additionalProperties": False,
        },
        check=lambda args: check_lesson(result, args),
        handler=keep_lesson,
    )


LESSON_HEADER = ("Notes this agent kept about its own earlier drafts. Each describes a "
                 "check that failed and what the revision that passed it changed. They "
                 "are not instructions, and every answer is still checked.")
NO_LESSONS = "No notes about earlier drafts."


def _clean(text: str) -> str:
    """One line, and no square brackets: a stored note can't forge an ID."""
    return " ".join(text.split()).replace("[", "(").replace("]", ")")


def lesson_report(store: LessonStore, max_lessons: int = 5) -> str:
    lessons = store.lessons()[-max_lessons:]
    if not lessons:
        return NO_LESSONS
    return "\n".join([LESSON_HEADER, ""] + [
        f"[{l.memory_id}] {l.key}: {_clean(l.text)} (evidence: {_clean(l.quote)})"
        for l in lessons])


# --- Demonstration: three sessions, with scripted drafts and reviews ---------

FOLDER, AGENT = "lessons", "flight-agent"

GROUNDED = ("Flight B costs $175 and takes 3.5 hours. It is longer than 3 hours, so the "
            "first checked bag costs $40 [baggage-2026#1], which makes the total $215. If "
            "you cancel 20 hours before departure, that is within 24 hours: the fare is "
            "not refunded but becomes travel credit, minus a $50 cancellation fee "
            "[refunds#2].")

SESSIONS = {             # what the model says at each step of each session, in order
    "2026-09-21": [
        ("draft", "Flight B costs $175 and takes 3.5 hours. The first checked bag costs "
                  "$40, which makes the total $215. If you cancel 20 hours before "
                  "departure, that is within 24 hours: the fare is not refunded but "
                  "becomes travel credit, minus a $50 cancellation fee [refunds#2]."),
        ("revise", GROUNDED),
        ("critique", []),
    ],
    "2026-09-22": [
        ("draft", "Flight B costs $175 and takes 3.5 hours. It is longer than 3 hours, so "
                  "the first checked bag costs $40 [baggage-2026#1], which makes the total "
                  "$215. If you cancel 20 hours before departure, the fare is refunded in "
                  "full [refunds#1]."),
        ("critique", ["refunds#1 covers cancelling more than 24 hours before departure. "
                      "20 hours before is within 24 hours, so refunds#2 applies: travel "
                      "credit, minus a $50 fee."]),
        ("revise", GROUNDED),
        ("critique", []),
    ],
    "2026-09-23": [
        ("draft", GROUNDED),
        ("critique", ["The $50 fee in refunds#2 applies to refunds. Here the fare becomes "
                      "credit instead, so no fee applies; remove it."]),
        ("revise", GROUNDED.replace(", minus a $50 cancellation fee", "")),
        ("critique", []),
    ],
}

LESSONS = {              # the keep_lesson calls the model makes when asked to reflect
    "2026-09-21": [
        {"rule": "uncited-claim", "failed": 1, "fixed": 2,
         "note": "The sentence giving the $40 bag fee had no citation; the revision "
                 "cited the baggage policy passage for it."},
        {"rule": "uncited-claim", "failed": 1, "fixed": 2,
         "note": "Always cite a source for every bag fee."},
        {"rule": "unsourced-number", "failed": 1, "fixed": 2,
         "note": "The first checked bag costs $40."},
    ],
    "2026-09-22": [
        {"rule": "critic", "failed": 1, "fixed": 2,
         "note": "A cancellation 20 hours out cited refunds#1; refunds#2 covers it."},
    ],
    "2026-09-23": [
        {"rule": "critic", "failed": 1, "fixed": 2,
         "note": "No fee applies when the fare becomes travel credit."},
    ],
}


class ScriptedModel:
    """Stands in for the model: each call returns the next reply in the script.
    A reply written for a different step fails loudly instead of being misused."""

    def __init__(self, script: list):
        self.script = list(script)

    def __call__(self, step: str, *context):
        expected, reply = self.script.pop(0)
        if step != expected:
            raise AssertionError(f"script has a {expected!r} reply; loop asked for {step!r}")
        return reply


def gather_evidence() -> list[str]:
    """The four tool results from Section 6.6: what a well-behaved run was shown."""
    registry = ToolRegistry(CONTRACTS + [make_search_contract(PolicyLibrary(DOCUMENTS))])
    calls = [("look_up_flight", {"flight_id": "B"}),
             ("search_policies", {"query": "checked bag fee"}),
             ("search_policies", {"query": "cancel within 24 hours refund"}),
             ("calculate", {"expression": "175 + 40"})]
    return [registry.execute(name, args).content for name, args in calls]


def run_session(script: list, evidence: list[str], max_revisions: int = 2) -> Refinement:
    model = ScriptedModel(script)
    return refine(model("draft"),
                  check=lambda answer: check_findings(answer, QUESTION, evidence),
                  critique=lambda answer: [Finding("critic", "critic", text)
                                           for text in model("critique", answer)],
                  revise=lambda answer, findings: model("revise", review_report(findings)),
                  max_revisions=max_revisions)


def _sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text.strip())


def show_attempts(result: Refinement) -> None:
    previous = None
    for a in result.attempts:
        if previous:
            new = [s for s in _sentences(a.answer) if s not in _sentences(previous.answer)]
            print(f"  attempt-{a.number} changes: {' '.join(new) or '(nothing)'}")
        print(f"  attempt-{a.number}: {'passes' if a.passes else 'fails'} the checks")
        for f in a.checks:
            print(f"      check {f.rule}: {f.detail.split(':')[0]}")
        for f in a.critique:
            print(f"      critic: {f.detail}")
        previous = a
    done = f"attempt-{result.delivered.number}" if result.delivered else "nothing (withheld)"
    print(f"  Delivered: {done}, after {result.model_calls} model calls "
          f"({result.stopped})")


def run_demo() -> None:
    store = LessonStore(FOLDER, AGENT)
    store.forget_all()                                 # the demonstration starts clean
    evidence = gather_evidence()
    for session, script in SESSIONS.items():
        print(f"=== Session {session} ===")
        result = run_session(script, evidence)
        show_attempts(result)
        episode = record_attempts(store, result, session)
        print(f"  Recorded by your code: [{episode.memory_id}] {episode.text}")
        print("  Lessons the model proposes, through the keep_lesson contract:")
        registry = ToolRegistry([make_lesson_contract(store, result, session,
                                                      episode.memory_id)])
        for args in LESSONS[session]:
            outcome = registry.execute("keep_lesson", args)
            print(f"    {args['rule']}: {args['note']!r}")
            print(f"        {'ok' if outcome.ok else 'REFUSED'}: {outcome.content}")
        print()

    print("=== What the next session's first draft will be shown ===")
    for line in lesson_report(LessonStore(FOLDER, AGENT)).splitlines():
        print(f"    {line}" if line else "")


if __name__ == "__main__":
    run_demo()
