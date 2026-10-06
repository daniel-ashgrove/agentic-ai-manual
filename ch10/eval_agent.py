"""
The harness around live agents: Section 6.7's one-call agent and Section 8.7's
critique agent, run on every case, with every model call and token counted, and
lessons judged on copies of the store before the real one is changed.
"""

import shutil
import tempfile
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage
from langchain_core.tracers.context import register_configure_hook

from critique_agent import EVIDENCE_TOOLS, CritiqueAgent                 # Section 8.7
from eval_harness import (CASES, ScoreLog, System, Trial, decide_lesson,
                          run_eval)
from policy_agent import (SYSTEM_PROMPT, ContractMiddleware,              # Section 6.7
                          as_langchain_tool, contracts, registry)
from policy_retrieval import check_answer
from self_critique import LessonStore

STATED_PROMPT = SYSTEM_PROMPT + (
    " Cite the passage for every fee you state. When you compare flights, add every "
    "fee that applies to each flight before you choose. When a traveller asks about "
    "cancelling, say what they get back and any fee that is deducted.")


# --- Counting every model call, at every level ------------------------------

class Meter(BaseCallbackHandler):
    """Counts every model call made while it is active, with the tokens each reports:
    the agent's tool-calling turns, its drafts and revisions, and the critic."""

    def __init__(self):
        super().__init__()
        self.calls, self.tokens, self._lock = 0, 0, threading.Lock()

    def on_llm_end(self, response, **kwargs) -> None:
        message = getattr(response.generations[0][0], "message", None)
        usage = getattr(message, "usage_metadata", None) or {}
        with self._lock:
            self.calls += 1
            self.tokens += usage.get("total_tokens", 0)


_active: ContextVar = ContextVar("eval_meter", default=None)
register_configure_hook(_active, inheritable=True)   # on every run inside `metered`


@contextmanager
def metered():
    meter = Meter()
    token = _active.set(meter)
    try:
        yield meter
    finally:
        _active.reset(token)


def _evidence(messages) -> tuple:
    return tuple(str(m.text) for m in messages if isinstance(m, ToolMessage)
                 and m.status == "success" and m.name in EVIDENCE_TOOLS)


# --- The systems under test --------------------------------------------------

def one_call_system(model, system_prompt: str = SYSTEM_PROMPT) -> System:
    """Section 6.7's agent: one answer, checked, delivered or withheld."""
    agent = create_agent(model=model, tools=[as_langchain_tool(c) for c in contracts],
                         system_prompt=system_prompt,
                         middleware=[ContractMiddleware(registry)])

    def system(case, n: int) -> Trial:
        with metered() as meter:
            messages = agent.invoke({"messages": [
                {"role": "user", "content": case.question}]})["messages"]
        answer, results = str(messages[-1].text), _evidence(messages)
        if check_answer(answer, case.question, list(results)):
            answer = None
        return Trial(case.case_id, n, answer, results, meter.calls, meter.tokens)
    return system


def critique_system(model, critic, store: LessonStore, run_id: str,
                    max_revisions: int = 2) -> System:
    """Section 8.7's agent, a fresh one for every trial, recording its episodes in
    `store`. Pass a copy: an evaluation should not write to the store the agent
    reads."""
    def system(case, n: int) -> Trial:
        agent = CritiqueAgent(model, critic, store, f"{run_id}/{case.case_id}/{n}",
                              max_revisions)
        with metered() as meter:
            agent.answer(case.question)
        done = agent.result.delivered
        return Trial(case.case_id, n, done.answer if done else None,
                     _evidence(agent.messages), meter.calls, meter.tokens)
    return system


# --- Lessons: judged on copies, applied only by decision ---------------------

def store_copy(store: LessonStore, without: tuple = ()) -> LessonStore:
    """The store as it is now, in a folder of its own, minus the lessons named."""
    folder = tempfile.mkdtemp(prefix="eval-")
    if store.path.exists():
        shutil.copy(store.path, Path(folder) / store.path.name)
    copy = LessonStore(folder, store.path.stem)
    for lesson_id in without:
        copy.forget(lesson_id)
    return copy


def judge_lessons(model, critic, store: LessonStore, log: ScoreLog, cases=CASES,
                  trials: int = 3) -> dict:
    """Run the agent with every current lesson, then without each one in turn, on
    every case. Returns a decision per lesson; nothing in `store` is changed."""
    def run(run_id, without=()):
        scores = run_eval(critique_system(model, critic, store_copy(store, without),
                                          run_id), cases, trials)
        log.record(run_id, {"system": "critique", "without": list(without)}, scores)
        return scores

    with_all = run("with-all-lessons")
    return {l.memory_id: decide_lesson(with_all, run(f"without-{l.memory_id}",
                                                     (l.memory_id,)))
            for l in store.lessons()}


def apply_decisions(store: LessonStore, decisions: dict) -> list[str]:
    """Retire what the harness decided to retire. 'review' waits for a person."""
    retired = [lesson for lesson, (decision, _) in decisions.items()
               if decision == "retire"]
    for lesson in retired:
        store.forget(lesson)
    return retired


if __name__ == "__main__":
    import os
    import sys

    from langchain_anthropic import ChatAnthropic

    MODEL = os.environ.get("BOOK_MODEL")                 # a current Claude model ID
    if not MODEL:
        sys.exit("BOOK_MODEL is not set. Set it to a current Claude model ID "
                 "and run this again.")
    model = ChatAnthropic(model=MODEL)
    log, store = ScoreLog("evals/live.jsonl"), LessonStore("lessons", "flight-agent")
    systems = {"one call": one_call_system(model),
               "one call, stated prompt": one_call_system(model, STATED_PROMPT),
               "critique loop": critique_system(model, model, store_copy(store),
                                                "critique")}
    for name, system in systems.items():
        entry = log.record(name, {"system": name, "model": MODEL},
                           run_eval(system, trials=5))
        print(f"{name}: correct {entry['correct']} of {entry['trials']}, withheld "
              f"{entry['withheld']}, {entry['model_calls']} model calls, "
              f"{entry['tokens']} tokens")
    decisions = judge_lessons(model, model, store, log)
    for lesson, (decision, reason) in decisions.items():
        print(f"{lesson}: {decision} ({reason})")
    print("retired:", apply_decisions(store, decisions) or "none")
