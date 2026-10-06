"""Tests the agent's wiring with a scripted stand-in for the model: the real
create_agent graph, middleware, tools and store run; only the model's replies are
fixed. Two sessions use two separate MemoryAgent and MemoryStore objects, so the
only thing they share is the file on disk. This checks the plumbing. It says
nothing about how a live model will behave.

Run: python -m unittest -v test_memory_agent   (needs langchain installed)"""

import logging
import tempfile
import unittest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import memory_agent as ma
from agent_memory import MemoryStore, make_memory_contracts, MemorySession

logging.getLogger("contracts").setLevel(logging.CRITICAL)


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


def call(name, args, i=0):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args,
                                              "id": f"call_{i}", "type": "tool_call"}])


GOOD_WRITE = {"key": "checked-bags", "fact": "Checks one bag on every trip.",
              "quote": "I always check a bag"}


def first_session(folder, write=GOOD_WRITE):
    model = ScriptedModel(script=[AIMessage(content="Flight B, at $175."),
                                  call("remember", write),
                                  AIMessage(content="Then flight A: $210, against $215 for B.")],
                          seen=[])
    agent = ma.MemoryAgent(model, MemoryStore(folder, "t-1041"), "2026-09-10")
    agent.ask("Which flight gets me there before 18:00 for the least money?")
    agent.ask("Wait, I always check a bag. Does that change it?")
    agent.close()
    return model


def second_session(folder, user="t-1041"):
    model = ScriptedModel(script=[AIMessage(content="Flight A: $210 with your bag.")], seen=[])
    agent = ma.MemoryAgent(model, MemoryStore(folder, user), "2026-09-17")
    agent.ask("Same trip next Friday, please.")
    return model


def recalled(model) -> str:
    """The recall report in the first request the model received this session."""
    return next(str(m.content) for m in model.seen[0] if isinstance(m, ToolMessage))


class MemoryWiring(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()

    def test_first_session_starts_with_no_notes(self):
        model = first_session(self.folder)
        self.assertIn("No notes from earlier sessions", recalled(model))

    def test_second_session_is_shown_the_first_sessions_fact(self):
        first_session(self.folder)
        report = recalled(second_session(self.folder))
        self.assertIn('[fact-1] Checks one bag on every trip. (said 2026-09-10: '
                      '"I always check a bag")', report)
        self.assertIn("[episode-1] 2026-09-10: Asked: Which flight gets me there", report)
        self.assertIn("Answered: Then flight A: $210, against $215 for B.", report)

    def test_recall_arrives_as_a_tool_result_not_in_the_system_prompt(self):
        first_session(self.folder)
        seen = second_session(self.folder).seen[0]
        system = [str(m.content) for m in seen if isinstance(m, SystemMessage)]
        self.assertEqual(system, [ma.SYSTEM_PROMPT])
        self.assertTrue(any(isinstance(m, ToolMessage) and "[fact-1]" in str(m.content)
                            for m in seen))

    def test_an_invented_quote_is_refused_and_nothing_is_kept(self):
        model = first_session(self.folder, write={**GOOD_WRITE, "quote": "I prefer aisle seats"})
        refusals = [m for m in model.seen[-1] if isinstance(m, ToolMessage) and m.status == "error"]
        self.assertEqual(len(refusals), 1)
        self.assertIn("does not appear in anything the traveller said", refusals[0].content)
        self.assertEqual(MemoryStore(self.folder, "t-1041").facts(), [])

    def test_another_traveller_is_shown_nothing(self):
        first_session(self.folder)
        self.assertIn("No notes from earlier sessions", recalled(second_session(self.folder, "t-2207")))

    def test_model_is_shown_the_contract_schemas(self):
        session = MemorySession(MemoryStore(self.folder, "t-1041"), "s")
        contracts = ma.CONTRACTS + make_memory_contracts(session)
        for contract in contracts:
            self.assertEqual(ma.as_langchain_tool(contract).args_schema, contract.input_schema)


if __name__ == "__main__":
    unittest.main()
