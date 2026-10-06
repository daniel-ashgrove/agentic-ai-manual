"""Tests the harness's wiring around real LangChain agents, with scripted stand-ins
for the model and the critic: the create_agent graphs, middleware, tools, checks,
loop, stores and meter run; only the models' replies are fixed. This checks the
plumbing. It says nothing about how a live model will score.

Run: python -m unittest -v test_eval_agent   (needs langchain installed)"""

import logging
import tempfile
import unittest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import eval_agent as ea
import eval_harness as eh
from self_critique import GROUNDED, SESSIONS, LessonStore

logging.getLogger("contracts").setLevel(logging.CRITICAL)
CASE = {c.case_id: c for c in eh.CASES}
USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}


def _calls(*pairs):
    return AIMessage(content="", usage_metadata=USAGE, tool_calls=[
        {"name": n, "args": a, "id": f"call_{i}", "type": "tool_call"}
        for i, (n, a) in enumerate(pairs)])


def _say(text):
    return AIMessage(content=text, usage_metadata=USAGE)


class Scripted(BaseChatModel):
    """Replies with the next message in its script."""
    script: list

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted"


class ReadsLessons(BaseChatModel):
    """Answers the flight A question, adding the $40 fee whenever lesson-2 is among the
    notes it was shown, which reach it only as the recall tool's result."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        notes = " ".join(str(m.content) for m in messages
                         if isinstance(m, ToolMessage) and m.name == "recall_lessons")
        swayed = "[lesson-2]" in notes
        if not any(isinstance(m, ToolMessage) and m.name == "look_up_flight" for m in messages):
            extra = [("calculate", {"expression": "210 + 40"})] if swayed else []
            reply = _calls(("look_up_flight", {"flight_id": "A"}),
                           ("search_policies", {"query": "checked bag fee"}), *extra)
        else:
            reply = _say(eh.BAG_A_WITH_FEE if swayed else eh.BAG_A)
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "reads-lessons"


class Approves(Scripted):
    script: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=_say("NO ISSUES"))])


FARE_B = [("look_up_flight", {"flight_id": "B"}), ("search_policies", {"query": "checked bag fee"}),
          ("search_policies", {"query": "cancel within 24 hours refund"}),
          ("calculate", {"expression": "175 + 40"})]
UNCITED = SESSIONS["2026-09-21"][0][1]


class Meter(unittest.TestCase):
    def test_every_call_at_every_level_is_counted(self):
        store = LessonStore(tempfile.mkdtemp(), "flight-agent")
        model = Scripted(script=[_calls(*FARE_B), _say(UNCITED), _say(GROUNDED)])
        critic = Scripted(script=[_say("NO ISSUES")])
        trial = ea.critique_system(model, critic, store, "A")(CASE["bag-and-cancel"], 1)
        self.assertEqual(trial.answer, GROUNDED)
        self.assertEqual((trial.model_calls, trial.tokens), (4, 480))   # the loop counts 3
        self.assertEqual(len(trial.tool_results), 4)

    def test_nothing_is_counted_outside_the_block(self):
        with ea.metered() as meter:
            Scripted(script=[_say("x")]).invoke("hi")
        Scripted(script=[_say("y")]).invoke("hi")
        self.assertEqual(meter.calls, 1)


class Systems(unittest.TestCase):
    def test_one_call_withholds_an_untraceable_answer(self):
        model = Scripted(script=[_calls(*FARE_B), _say(UNCITED)])
        trial = ea.one_call_system(model)(CASE["bag-and-cancel"], 1)
        self.assertIsNone(trial.answer)
        self.assertEqual((trial.model_calls, len(trial.tool_results)), (2, 4))

    def test_one_call_delivers_a_traceable_answer(self):
        model = Scripted(script=[_calls(*FARE_B), _say(GROUNDED)])
        s = eh.score_trial(CASE["bag-and-cancel"], ea.one_call_system(model)(CASE["bag-and-cancel"], 1))
        self.assertEqual((s.outcome, s.traceable), ("correct", True))

    def test_the_stated_prompt_is_the_one_the_agent_gets(self):
        seen = []

        class Records(Scripted):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                seen.append(str(messages[0].content))
                return super()._generate(messages)

        ea.one_call_system(Records(script=[_say("x")]), ea.STATED_PROMPT)(CASE["bag-A"], 1)
        self.assertIn("add every fee that applies", seen[0])

    def test_lesson_notes_are_not_evidence(self):
        store = LessonStore(tempfile.mkdtemp(), "flight-agent")
        eh.keep_chapter_8_lessons(store)
        trial = ea.critique_system(ReadsLessons(), Approves(), store, "A")(CASE["bag-A"], 1)
        self.assertFalse(any("Notes this agent kept" in r for r in trial.tool_results))


class Lessons(unittest.TestCase):
    def setUp(self):
        self.store = LessonStore(tempfile.mkdtemp(), "flight-agent")
        eh.keep_chapter_8_lessons(self.store)

    def test_a_copy_can_drop_a_lesson_without_touching_the_store(self):
        copy = ea.store_copy(self.store, without=("lesson-2",))
        self.assertEqual([l.memory_id for l in copy.lessons()], ["lesson-1"])
        self.assertEqual(len(self.store.lessons()), 2)

    def test_a_trial_records_its_episode_in_the_copy_only(self):
        before = len(self.store.episodes())
        copy = ea.store_copy(self.store)
        ea.critique_system(ReadsLessons(), Approves(), copy, "A")(CASE["bag-A"], 1)
        self.assertEqual((len(self.store.episodes()), len(copy.episodes())), (before, before + 1))

    def test_lessons_are_judged_and_applied_only_by_decision(self):
        log = eh.ScoreLog(tempfile.mkdtemp() + "/runs.jsonl")
        decisions = ea.judge_lessons(ReadsLessons(), Approves(), self.store, log,
                                     cases=(CASE["bag-A"],), trials=2)
        self.assertEqual({k: v[0] for k, v in decisions.items()},
                         {"lesson-1": "keep", "lesson-2": "retire"})
        self.assertEqual(len(self.store.lessons()), 2)            # judging changed nothing
        self.assertEqual(ea.apply_decisions(self.store, decisions), ["lesson-2"])
        self.assertEqual([l.memory_id for l in self.store.lessons()], ["lesson-1"])
        self.assertEqual([r["run"] for r in log.runs()],
                         ["with-all-lessons", "without-lesson-1", "without-lesson-2"])

    def test_review_is_left_for_a_person(self):
        self.assertEqual(ea.apply_decisions(self.store, {"lesson-1": ("review", ""),
                                                         "lesson-2": ("keep", "")}), [])
        self.assertEqual(len(self.store.lessons()), 2)


if __name__ == "__main__":
    unittest.main()
