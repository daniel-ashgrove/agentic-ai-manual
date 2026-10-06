"""Tests the guardrail layer's wiring around real agent graphs, with scripted
stand-ins for every model. The create_agent graphs, middleware, contracts, checks,
limits and halt conditions all run; only the models' replies are fixed.

Run: python -m unittest -v test_guarded_agent   (needs langchain installed)
"""

import logging
import unittest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import guarded_agent as ga
import guardrails as g

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


CREDIT = {"booking": "BK-4471", "amount_usd": 125,
          "reason": "the $175 fare minus the $50 cancellation fee [refunds#2]"}
FARE_REPORT = ("Flight B's fare is $175 and it takes 3.5 hours, so the first checked bag "
               "cost $40 [baggage-2026#1].")
RECORDED = "Recorded a request for $125 of travel credit on BK-4471."


def orchestra(coordinator, ledger=None, limits=g.Limits(), **workers):
    models = {name: scripted(*replies) for name, replies in workers.items()}
    for name in list(g.WORKERS) + ["service"]:
        models.setdefault(name, scripted("nothing to report"))
    return ga.GuardedOrchestra(scripted(*coordinator), models,
                               ledger or g.Ledger(), limits)


def delegate(worker, task):
    return ("delegate", {"worker": worker, "task": task})


class Wiring(unittest.TestCase):

    def test_the_coordinator_gets_delegate_and_no_acting_tool(self):
        system = orchestra(["Nothing to do here."])
        system.answer("hello?", "T-1", "2026-06-02")
        self.assertEqual(system.model.tools, [["delegate"]])

    def test_only_the_service_worker_can_ask_for_money(self):
        system = orchestra(
            [calls(delegate("service", "Request $125 travel credit on BK-4471.")),
             "I've asked for $125 of credit on BK-4471."],
            service=[calls(("request_credit", CREDIT)), RECORDED],
            fares=[FARE_REPORT], policy=["nothing to report"])
        system.answer("cancel BK-4471; I paid $125", "T-1", "2026-06-02")
        tools = {name: model.tools for name, model in system.worker_models.items()}
        self.assertEqual({name for bound in tools["service"] for name in bound},
                         {"request_credit"})
        self.assertEqual(tools["fares"], [])       # never delegated to, never built
        self.assertNotIn("request_credit",
                         [name for worker in ("fares", "policy")
                          for bound in tools[worker] for name in bound])


class TheLayerAroundRealGraphs(unittest.TestCase):

    def test_a_request_is_held_and_the_ledger_is_untouched(self):
        ledger = g.Ledger()
        system = orchestra(
            [calls(delegate("service", "Request $125 travel credit on BK-4471.")),
             "I've asked for $125 of credit on BK-4471."],
            ledger=ledger,
            service=[calls(("request_credit", CREDIT)), RECORDED])
        run, outcome = system.answer("cancel BK-4471; I paid $125", "T-1", "2026-06-02")
        self.assertEqual([(a.booking, a.amount_usd) for a in run.held],
                         [("BK-4471", 125)])
        self.assertEqual(ledger.applied, [])
        g.review(run.held[0], "dana", ledger, approve=True)
        self.assertEqual(len(ledger.applied), 1)

    def test_an_untraceable_amount_halts_a_real_run(self):
        system = orchestra(
            [calls(delegate("service", "Request $85 travel credit on BK-4471.")),
             "I've asked for $85 of credit on BK-4471."],
            service=[calls(("request_credit", dict(CREDIT, amount_usd=85))), RECORDED])
        run, outcome = system.answer("cancel BK-4471", "T-1", "2026-06-02")
        self.assertEqual(outcome.halted, "untraceable-amount")
        self.assertIn("nothing has been charged", outcome.answer)

    def test_the_call_limit_refuses_a_delegation_mid_run(self):
        system = orchestra(
            [calls(delegate("fares", "Price flight B with one checked bag.")),
             calls(delegate("policy", "What happens on a cancellation?")),
             "I could not finish this."],
            limits=g.Limits(model_calls=2),
            fares=[calls(("look_up_flight", {"flight_id": "B"})), FARE_REPORT])
        run, outcome = system.answer("what do I get back?", "T-1", "2026-06-02")
        self.assertTrue(any("no model calls left" in line for line in run.log))
        self.assertEqual(outcome.halted, "budget-spent")

    def test_the_record_survives_a_halt(self):
        system = orchestra(
            [calls(delegate("service", "Request $85 travel credit on BK-4471.")),
             "I've asked for $85."],
            service=[calls(("request_credit", dict(CREDIT, amount_usd=85))), RECORDED])
        run, outcome = system.answer("cancel BK-4471", "T-1", "2026-06-02")
        packet = g.escalation(run, outcome)
        self.assertIn("not issued", packet)
        self.assertIn("service report", packet)


class TheFrameworksOwnCheckpoint(unittest.TestCase):

    def build(self):
        run, ledger = g.Run("T-1", "cancel BK-4471", "2026-06-02"), g.Ledger()
        contract = g.make_credit_contract(run, ledger)
        model = scripted(calls(("request_credit", CREDIT)), RECORDED)
        agent, config = ga.paused_service(model, contract)
        return run, agent, config

    def test_the_tool_does_not_run_until_the_graph_is_resumed(self):
        run, agent, config = self.build()
        state = agent.invoke({"messages": [HumanMessage("Request $125 on BK-4471.")]},
                             config)
        self.assertEqual(run.held, [])
        requested = ga.pending(state)
        self.assertEqual(requested[0]["name"], "request_credit")
        self.assertEqual(requested[0]["args"]["amount_usd"], 125)

    def test_approving_runs_the_tool_exactly_once(self):
        run, agent, config = self.build()
        agent.invoke({"messages": [HumanMessage("Request $125 on BK-4471.")]}, config)
        agent.invoke(ga.Command(resume=ga.APPROVE), config)
        self.assertEqual([(a.booking, a.amount_usd) for a in run.held],
                         [("BK-4471", 125)])

    def test_rejecting_leaves_the_request_unrecorded(self):
        run, agent, config = self.build()
        agent.invoke({"messages": [HumanMessage("Request $125 on BK-4471.")]}, config)
        agent.invoke(ga.Command(resume=ga.REJECT), config)
        self.assertEqual(run.held, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
