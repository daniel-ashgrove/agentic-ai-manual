"""Tests the service around real agent graphs, with scripted stand-ins for every
model. Service, EventLog and the dashboard are the same code as in test_deploy;
only the engine changes, which is the claim this file exists to check.

Run: python -m unittest -v test_deploy_agent   (needs langchain installed)
"""

import logging
import tempfile
import unittest

import dashboard as w
import deploy as d
import deploy_agent as da
from guardrails import Ledger
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from orchestration import WORKERS

logging.getLogger("contracts").setLevel(logging.CRITICAL)


class ScriptedModel(BaseChatModel):
    """Replies with the next message in its script, as in Chapter 12's tests."""
    script: list

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted"


def scripted(*replies):
    return ScriptedModel(script=[r if isinstance(r, AIMessage) else AIMessage(content=r)
                                 for r in replies])


def calls(*pairs):
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": args, "id": f"call_{i}_{name}", "type": "tool_call"}
        for i, (name, args) in enumerate(pairs)])


ASK = "What do I get back if I cancel 20 hours out?"
RULE = ("Cancelling within 24 hours of departure means the fare becomes travel credit "
        "minus a $50 cancellation fee, valid for 12 months [refunds#2].")
POLICY = [calls(("search_policies",
                 {"query": "cancel within 24 hours refund checked bag fees"})), RULE]


def engine_for(coordinator, policy):
    workers = {name: scripted(*policy) for name in list(WORKERS) + ["service"]}
    return da.agent_engine(scripted(*coordinator), workers, scripted("NO ISSUES"))


class TheSameServiceAroundRealAgents(unittest.TestCase):

    def service(self, folder, engine, **kwargs):
        clock = d.StepClock()
        log = d.EventLog(f"{folder}/events.jsonl", da.LIVE, clock)
        return d.Service(folder, log, Ledger(), engine, ids=d.counting_ids(),
                         now=clock, **kwargs), log, clock

    def test_a_graph_run_produces_the_same_events_as_a_scripted_one(self):
        coordinator = [calls(("delegate", {"worker": "policy", "task":
                                           "What does a traveller get back when "
                                           "cancelling 20 hours before departure?"})),
                       RULE]
        with tempfile.TemporaryDirectory() as folder:
            service, log, _ = self.service(folder, engine_for(coordinator, POLICY))
            reply = service.answer(d.Request(ASK, "T-41", d.DAY_ONE))
            self.assertTrue(reply.delivered)
            self.assertEqual([e["event"] for e in log.read()],
                             ["received", "answered"])

    def test_the_dashboard_reads_an_agent_runs_log_unchanged(self):
        coordinator = [calls(("delegate", {"worker": "policy", "task":
                                           "What does a traveller get back when "
                                           "cancelling 20 hours before departure?"})),
                       RULE]
        with tempfile.TemporaryDirectory() as folder:
            service, log, _ = self.service(folder, engine_for(coordinator, POLICY))
            service.answer(d.Request(ASK, "T-41", d.DAY_ONE))
            figures = w.summarise(log.read())
            self.assertEqual((figures["runs"], figures["delivered"]), (1, 1))
            self.assertEqual(figures["releases"],
                             [f"flight-agent 1.0.0 / {da.LIVE.model}"])

    def test_a_request_already_past_its_deadline_is_not_run_at_all(self):
        with tempfile.TemporaryDirectory() as folder:
            service, log, _ = self.service(
                folder, engine_for([RULE], POLICY), deadline_s=-1.0)
            reply = service.answer(d.Request(ASK, "T-41", d.DAY_ONE))
            self.assertFalse(reply.delivered)
            self.assertEqual(reply.answer, d.SORRY)
            self.assertEqual([e["event"] for e in log.read()],
                             ["received", "deadline", "failed"])

    def test_the_release_is_the_one_the_events_name(self):
        self.assertEqual((da.LIVE.provider, da.LIVE.service), ("anthropic",
                                                               "flight-agent"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
