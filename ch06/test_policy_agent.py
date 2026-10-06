"""Tests the agent's wiring with a scripted stand-in for the model: the real
create_agent graph, middleware, and tools run; only the model's replies are fixed.
This checks the plumbing. It says nothing about how a live model will behave.

Run: python -m unittest -v test_policy_agent   (needs langchain installed)"""

import logging
import unittest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import policy_agent as pa
from policy_retrieval import CANDIDATE_ANSWERS, QUESTION

GROUNDED = dict(CANDIDATE_ANSWERS)["grounded"]
logging.getLogger("contracts").setLevel(logging.CRITICAL)   # withheld answers are expected here


class ScriptedModel(BaseChatModel):
    """Replies with the next message in its script, and records what it was sent."""
    script: list
    seen: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted"


def calls(*pairs):
    return [AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"call_{i}",
                                               "type": "tool_call"}])
            for i, (n, a) in enumerate(pairs)]


EVIDENCE = calls(("look_up_flight", {"flight_id": "B"}),
                 ("search_policies", {"query": "checked bag fee"}),
                 ("search_policies", {"query": "cancel within 24 hours refund"}),
                 ("calculate", {"expression": "175 + 40"}))


def run(script):
    model = ScriptedModel(script=list(script), seen=[])
    return pa.respond(pa.build_agent(model), QUESTION), model


def tool_messages(model):
    return [m for m in model.seen[-1] if isinstance(m, ToolMessage)]


class AgentWiring(unittest.TestCase):
    def test_grounded_answer_is_returned(self):
        answer, _ = run(EVIDENCE + [AIMessage(content=GROUNDED)])
        self.assertEqual(answer, GROUNDED)

    def test_uncited_answer_is_withheld(self):
        answer, _ = run(EVIDENCE + [AIMessage(content=dict(CANDIDATE_ANSWERS)["uncited"])])
        self.assertEqual(answer, pa.WITHHELD)

    def test_answer_with_no_searches_is_withheld(self):
        answer, _ = run([AIMessage(content=GROUNDED)])
        self.assertEqual(answer, pa.WITHHELD)

    def test_gates_refuse_before_any_tool_runs(self):
        script = calls(("search_policies", {"query": "fee", "scope": "community"}),
                       ("search_policies", {"query": "what is it?"}),
                       ("calculate", {"expression": "1 / (2 - 2)"}),
                       ("book_flight", {"flight_id": "B"}))
        _, model = run(script + [AIMessage(content="I could not answer.")])
        seen = tool_messages(model)
        self.assertEqual([m.status for m in seen], ["error"] * 4)
        self.assertIn("unexpected argument 'scope'", seen[0].content)
        self.assertIn("no searchable words", seen[1].content)
        self.assertIn("division by zero", seen[2].content)
        self.assertIn("Unknown tool 'book_flight'", seen[3].content)

    def test_model_is_shown_the_registry_schemas(self):
        tools = {t.name: t for t in (pa.as_langchain_tool(c) for c in pa.contracts)}
        for contract in pa.contracts:
            self.assertEqual(tools[contract.name].args_schema, contract.input_schema)


if __name__ == "__main__":
    unittest.main()
