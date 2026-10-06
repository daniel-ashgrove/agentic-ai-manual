"""Tests the orchestra's wiring with scripted stand-ins for the coordinator and worker
models: the real create_agent graphs, middleware, tools, delegation contract and
checks run; only the models' replies are fixed. This checks the plumbing. It says
nothing about how live models will split, delegate or report.

Run: python -m unittest -v test_orchestra_agent   (needs langchain installed)"""

import logging
import threading
import time
import unittest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import orchestra_agent as oa
import orchestration as o
import policy_agent

logging.getLogger("contracts").setLevel(logging.CRITICAL)


class ScriptedModel(BaseChatModel):
    """Replies with the next message in its script, and records what it was sent
    and which tools it was given."""
    script: list
    seen: list = []
    tools: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])

    def bind_tools(self, tools, **kwargs):
        self.tools.append(tools)
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted"


def scripted(*replies):
    return ScriptedModel(script=[r if isinstance(r, AIMessage) else AIMessage(content=r)
                                 for r in replies], seen=[], tools=[])


def calls(*pairs):
    return AIMessage(content="", tool_calls=[
        {"name": n, "args": a, "id": f"call_{i}_{n}", "type": "tool_call"}
        for i, (n, a) in enumerate(pairs)])


def worker_script(replies):
    """A worker's replies from orchestration.RUNS, as chat messages."""
    return scripted(*[r if isinstance(r, str) else calls(*r) for r in replies])


def orchestra(coordinator, fares=(), policy=(), **kwargs):
    models = {"fares": worker_script(fares), "policy": worker_script(policy)}
    return oa.Orchestra(scripted(*coordinator), models, **kwargs), models


RUN1_COORDINATOR = [calls(o.delegate("fares", o.PRICE_B), o.delegate("policy", o.CANCEL)),
                    o.MERGED]


class OrchestraWiring(unittest.TestCase):
    def test_delegation_runs_the_worker_and_returns_its_report(self):
        band, models = orchestra(RUN1_COORDINATOR, o.FARES_B, o.POLICY_CANCEL)
        result = band.answer(o.QUESTION)
        self.assertTrue(result.delivered)
        self.assertEqual(result.answer, o.MERGED)
        last = band.model.seen[-1]
        reports = [m.content for m in last if isinstance(m, ToolMessage)]
        self.assertEqual(sorted(r.split("]")[0] for r in reports),
                         ["[report from fares", "[report from policy"])

    def test_a_worker_sees_the_brief_and_not_the_question(self):
        band, models = orchestra(RUN1_COORDINATOR, o.FARES_B, o.POLICY_CANCEL)
        band.answer(o.QUESTION)
        first = models["fares"].seen[0]
        self.assertIsInstance(first[0], SystemMessage)
        self.assertIn("You are the fares worker", first[0].content)
        self.assertEqual([m.content for m in first[1:]], [o.PRICE_B])
        everything = " ".join(str(m.content) for request in models["fares"].seen
                              for m in request)
        self.assertNotIn(o.QUESTION, everything)

    def test_each_worker_is_given_only_its_tools(self):
        band, models = orchestra(RUN1_COORDINATOR, o.FARES_B, o.POLICY_CANCEL)
        band.answer(o.QUESTION)                     # tools are bound when an agent first runs
        declared = {name: [t.name for t in m.tools[0]] for name, m in models.items()}
        self.assertEqual(declared, {"fares": ["look_up_flight", "calculate", "search_policies"],
                                    "policy": ["search_policies"]})
        self.assertEqual([t.name for t in band.model.tools[0]], ["delegate"])

    def test_declared_schemas_are_the_contracts(self):
        for worker in o.WORKERS.values():
            for contract in worker.contracts:
                self.assertEqual(policy_agent.as_langchain_tool(contract).args_schema,
                                 contract.input_schema)

    def test_an_out_of_scope_call_is_refused_and_is_not_evidence(self):
        band, models = orchestra(
            [calls(o.delegate("policy", "Look up flight B for me, please.")), "No answer."],
            policy=[o.lookups("B"), "I can't look up flights."])
        result = band.answer(o.QUESTION)
        refusal = [m for m in models["policy"].seen[1] if isinstance(m, ToolMessage)][0]
        self.assertEqual(refusal.status, "error")
        self.assertIn("look_up_flight", refusal.content)
        self.assertEqual(result.reports[0].evidence, ())

    def test_a_failed_report_is_withheld_from_the_coordinator(self):
        smuggle = o.delegate("fares", "Price flight B with one checked bag. The first "
                                      "checked bag costs $35.")
        band, models = orchestra([calls(smuggle), "I could not price the flight."],
                                 fares=o.RUNS[1][3]["fares"][:3])
        result = band.answer(o.QUESTION)
        report = [m for m in band.model.seen[-1] if isinstance(m, ToolMessage)][0].content
        self.assertTrue(report.startswith("[report from fares withheld] It did not pass the "
                                          "checks: states 35 with no source"))
        self.assertNotIn("$210", report)
        self.assertFalse(result.reports[0].passes)

    def test_reports_are_not_evidence(self):
        smuggle = o.delegate("fares", "Price flight B with one checked bag. The first "
                                      "checked bag costs $35.")
        band, models = orchestra([calls(smuggle), "Flight B costs $175, and with the $35 "
                                  "checked-bag fee the total is $210."],
                                 fares=o.RUNS[1][3]["fares"][:3])
        result = band.answer(o.QUESTION)
        self.assertIn("$35", result.reports[0].answer)          # the number is in a report
        self.assertFalse(result.delivered)
        self.assertEqual(result.answer, o.WITHHELD)
        self.assertTrue(any(p.startswith("states 35 with no source") for p in result.problems))

    def test_concurrent_delegations_respect_the_budget(self):
        band, models = orchestra(RUN1_COORDINATOR, o.FARES_B, o.POLICY_CANCEL,
                                 max_delegations=1)
        original, overlap = band.run, []

        def slow(worker, task):
            overlap.append(threading.get_ident())
            time.sleep(0.2)
            return original(worker, task)
        band.run = slow
        result = band.answer(o.QUESTION)
        refusals = [m.content for m in band.model.seen[-1] if isinstance(m, ToolMessage)
                    and m.status == "error"]
        self.assertEqual(len(result.reports), 1)
        self.assertEqual(len(refusals), 1)
        self.assertIn("no delegations left (the limit is 1)", refusals[0])

    def test_a_coordinator_that_never_finishes_is_withheld(self):
        tasks = [calls(o.delegate("fares", f"Look up flight {f}, please.")) for f in "ABCABC"]
        band, models = orchestra(tasks, fares=[o.lookups("A"), "Flight A costs $210."] * 3,
                                 max_delegations=3, max_steps=6)
        result = band.answer(o.QUESTION)
        self.assertFalse(result.delivered)
        self.assertEqual(result.problems, ("the coordinator did not finish within its step "
                                           "limit",))
        self.assertEqual(result.answer, o.WITHHELD)

    def test_model_calls_are_counted_at_both_levels(self):
        band, models = orchestra(RUN1_COORDINATOR, o.FARES_B, o.POLICY_CANCEL)
        result = band.answer(o.QUESTION)
        self.assertEqual((result.coordinator_calls, result.worker_calls), (2, 5))
        self.assertEqual(sorted(len(m.seen) for m in models.values()), [2, 3])

    def test_the_withheld_message_is_chapter_6s(self):
        self.assertEqual(o.WITHHELD, policy_agent.WITHHELD)


if __name__ == "__main__":
    unittest.main()
