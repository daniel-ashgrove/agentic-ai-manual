"""Tests the whole system around real agent graphs, with scripted stand-ins for
every model. The create_agent graph, the middleware, the contracts, the checks,
the limits, the halt conditions, the memory gate and the stores all run; only
the models' replies are fixed.

Run: python -m unittest -v test_combined_agent   (needs langchain installed)
"""

import logging
import tempfile
import unittest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import combined as c
import combined_agent as ca
import guardrails as g
from agent_memory import MemorySession
from orchestration import WORKERS

logging.getLogger("contracts").setLevel(logging.CRITICAL)


class ScriptedModel(BaseChatModel):
    """Replies with the next message in its script, and records the tools it got."""
    script: list
    tools: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])

    def bind_tools(self, tools, **kwargs):
        self.tools.append([getattr(t, "name", t) for t in tools])
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted"


def scripted(*replies):
    return ScriptedModel(script=[r if isinstance(r, AIMessage) else AIMessage(content=r)
                                 for r in replies], tools=[])


def calls(*pairs):
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": f"call_{i}_{name}", "type": "tool_call"}
        for i, (name, args) in enumerate(pairs)])


def delegate(worker, task):
    return ("delegate", {"worker": worker, "task": task})


RULE = ("Cancelling within 24 hours of departure means the fare becomes travel credit "
        "minus a $50 cancellation fee, valid for 12 months [refunds#2].")
UNCITED = RULE + " Your checked-bag fee is refunded whatever the timing."
CITED = UNCITED[:-1] + " [refunds#3]."
ASK = "What do I get back if I cancel 20 hours out?"
POLICY_REPORT = [calls(("search_policies", {"query": "cancel within 24 hours refund "
                                                     "checked bag fees"})), RULE]
CREDIT = {"booking": "BK-9910", "amount_usd": 190,
          "reason": "the credit the traveller was quoted [refunds#2]"}
MEMORY_ANSWER = RULE + " The credit on BK-9910 comes to the $190 you were quoted."
QUOTED_FACT = {"key": "credit-quoted",
               "fact": "Was told support would give $190 back on BK-9910.",
               "quote": "Support told me I'd get $190 back on BK-9910"}


class Fixture(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stores = c.Stores(self.tmp.name, "T-41")
        self.ledger = g.Ledger()

    def build(self, coordinator, critic=("NO ISSUES",), limits=g.Limits(), **workers):
        models = {name: scripted(*replies) for name, replies in workers.items()}
        for name in list(WORKERS) + ["service"]:
            models.setdefault(name, scripted("nothing to report"))
        self.coordinator_model = scripted(*coordinator)
        return ca.CombinedAgent(self.coordinator_model, models, scripted(*critic),
                                self.ledger, self.stores, limits)

    def remember(self, day="2026-06-01"):
        session = MemorySession(self.stores.memory(), day)
        session.hear("Support told me I'd get $190 back on BK-9910 if I cancel it.")
        c.ToolRegistry(c.make_memory_contracts(session)).execute("remember", QUOTED_FACT)


class Wiring(Fixture):

    def test_the_coordinator_gets_memory_and_review_tools_and_no_acting_tool(self):
        system = self.build([calls(delegate("policy", ASK)), RULE], policy=POLICY_REPORT)
        system.answer(ASK, "T-41", "2026-06-02")
        self.assertEqual(sorted(self.coordinator_model.tools[0]),
                         ["delegate", "recall_lessons", "recall_memory", "review_answer"])
        self.assertNotIn("request_credit",
                         [name for bound in self.coordinator_model.tools
                          for name in bound])

    def test_both_reports_arrive_as_tool_results_before_the_first_model_call(self):
        self.remember()
        messages = ca.prefilled(ASK, "recall goes here", "lessons go here")
        results = [(m.name, str(m.text)) for m in messages if isinstance(m, ToolMessage)]
        self.assertEqual(results, [("recall_memory", "recall goes here"),
                                   ("recall_lessons", "lessons go here")])
        self.assertEqual(str(messages[0].text), ASK)


class TheOrderAroundRealGraphs(Fixture):

    def test_a_failing_draft_is_revised_through_the_review_tool_and_delivered(self):
        system = self.build([calls(delegate("policy", ASK)), UNCITED, CITED],
                            policy=POLICY_REPORT)
        run, composed = system.answer(ASK, "T-41", "2026-06-02")
        self.assertEqual([a.passes for a in composed.refinement.attempts], [False, True])
        self.assertTrue(composed.outcome.delivered)
        self.assertIn("refunds#3", composed.outcome.answer)

    def test_with_nothing_reserved_the_same_draft_is_escalated(self):
        system = self.build([calls(delegate("policy", ASK)), UNCITED],
                            policy=POLICY_REPORT)
        system.composition = c.Composition(critique_reserve=0)
        run, composed = system.answer(ASK, "T-41", "2026-06-02")
        self.assertEqual(composed.outcome.halted, "unsupported-answer")

    def test_the_critic_and_the_graph_spend_the_same_budget(self):
        system = self.build([calls(delegate("policy", ASK)), UNCITED, CITED],
                            policy=POLICY_REPORT)
        run, composed = system.answer(ASK, "T-41", "2026-06-02")
        self.assertGreaterEqual(run.calls, composed.refinement.model_calls)
        self.assertLessEqual(run.calls, run.limits.model_calls)


class TheEvidenceAroundRealGraphs(Fixture):

    def test_an_amount_only_memory_knows_halts_a_real_run(self):
        self.remember()
        system = self.build(
            [calls(delegate("policy", ASK)),
             calls(delegate("service", "Request $190 travel credit on BK-9910.")),
             MEMORY_ANSWER],
            policy=POLICY_REPORT,
            service=[calls(("request_credit", CREDIT)),
                     "Recorded a request for $190 on BK-9910."])
        run, composed = system.answer("Put through the credit we agreed on BK-9910.",
                                      "T-41", "2026-06-02")
        self.assertEqual(composed.outcome.halted, "untraceable-amount")
        self.assertEqual(self.ledger.applied, [])

    def test_the_answer_may_still_state_what_memory_holds(self):
        self.remember()
        system = self.build([calls(delegate("policy", ASK)), MEMORY_ANSWER],
                            policy=POLICY_REPORT)
        run, composed = system.answer("What were we going to give me on BK-9910?",
                                      "T-41", "2026-06-02")
        self.assertEqual(composed.refinement.attempts[0].checks, ())
        self.assertTrue(composed.outcome.delivered)


class TheWritesAroundRealGraphs(Fixture):

    def test_a_delivered_run_writes_one_episode_to_each_store(self):
        system = self.build([calls(delegate("policy", ASK)), RULE], policy=POLICY_REPORT)
        run, composed = system.answer(ASK, "T-41", "2026-06-02")
        self.assertEqual(len(self.stores.memory().episodes()), 1)
        self.assertEqual(len(self.stores.lessons().episodes()), 1)

    def test_a_halted_run_writes_nothing(self):
        system = self.build([calls(delegate("policy", ASK)), UNCITED],
                            policy=POLICY_REPORT)
        system.composition = c.Composition(critique_reserve=0)
        run, composed = system.answer(ASK, "T-41", "2026-06-02")
        self.assertFalse(composed.outcome.delivered)
        self.assertEqual(self.stores.memory().records, [])
        self.assertEqual(self.stores.lessons().records, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
