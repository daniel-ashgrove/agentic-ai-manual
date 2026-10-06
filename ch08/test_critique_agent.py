"""Tests the agent's wiring with scripted stand-ins for the model, the critic and
the reflector: the real create_agent graph, middleware, tools, check, loop and store
run; only the models' replies are fixed. This checks the plumbing. It says nothing
about how a live model will draft, critique or reflect.

Run: python -m unittest -v test_critique_agent   (needs langchain installed)"""

import logging
import tempfile
import unittest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import critique_agent as ca
import self_critique as sc

logging.getLogger("contracts").setLevel(logging.CRITICAL)
QUESTION = sc.QUESTION
UNCITED = sc.SESSIONS["2026-09-21"][0][1]
MISGROUNDED = sc.SESSIONS["2026-09-22"][0][1]


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


def scripted(*replies):
    return ScriptedModel(script=list(replies), seen=[])


LOOKUPS = AIMessage(content="", tool_calls=[
    {"name": n, "args": a, "id": f"call_{i}", "type": "tool_call"} for i, (n, a) in enumerate([
        ("look_up_flight", {"flight_id": "B"}),
        ("search_policies", {"query": "checked bag fee"}),
        ("search_policies", {"query": "cancel within 24 hours refund"}),
        ("calculate", {"expression": "175 + 40"})])])


def session(folder, drafts, critiques, max_revisions=2, day="2026-09-21"):
    model = scripted(LOOKUPS, *[AIMessage(content=d) for d in drafts])
    critic = scripted(*[AIMessage(content=c) for c in critiques])
    agent = ca.CritiqueAgent(model, critic, sc.LessonStore(folder, "flight-agent"), day,
                             max_revisions)
    return agent, agent.answer(QUESTION), model, critic


class CritiqueWiring(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()

    def test_failed_check_is_revised_through_a_review_tool_result(self):
        agent, answer, model, critic = session(self.folder, [UNCITED, sc.GROUNDED], ["NO ISSUES"])
        self.assertEqual(answer, sc.GROUNDED)
        last_request = model.seen[-1]
        reviewed = [m for m in last_request if isinstance(m, AIMessage) and m.content == UNCITED]
        self.assertEqual(reviewed[0].tool_calls[0]["name"], "review_answer")
        review = [m for m in last_request if isinstance(m, ToolMessage) and m.name == "review_answer"]
        self.assertIn("- (check: uncited-claim) makes a claim with no citation", review[0].content)
        self.assertEqual((len(critic.seen), agent.result.model_calls), (1, 3))

    def test_critic_gets_a_fresh_request_with_the_evidence(self):
        agent, answer, model, critic = session(self.folder, [MISGROUNDED, sc.GROUNDED],
                                               ["- refunds#1 does not cover 20 hours.", "NO ISSUES"])
        self.assertEqual(answer, sc.GROUNDED)
        first = critic.seen[0]
        self.assertEqual([m.content for m in first if isinstance(m, SystemMessage)], [ca.CRITIC_PROMPT])
        self.assertIn("[refunds#1] Cancellations and refunds", first[-1].content)
        self.assertIn(MISGROUNDED, first[-1].content)
        self.assertEqual(agent.result.attempts[0].critique[0].detail, "refunds#1 does not cover 20 hours.")

    def test_notes_and_reviews_are_not_evidence(self):
        store = sc.LessonStore(self.folder, "flight-agent")
        store.add_lesson("unsourced-number", "The first checked bag costs $40.", "ev", "s0")
        agent, answer, model, critic = session(self.folder, [UNCITED], [], max_revisions=0)
        self.assertIn("40", next(m.content for m in model.seen[0] if isinstance(m, ToolMessage)))
        self.assertEqual(answer, ca.WITHHELD)
        self.assertEqual({f.rule for f in agent.result.attempts[0].checks},
                         {"unsourced-number", "uncited-claim"})

    def test_nothing_passing_is_withheld_and_recorded(self):
        agent, answer, model, critic = session(self.folder, [UNCITED, UNCITED], [], max_revisions=1)
        self.assertEqual(answer, ca.WITHHELD)
        self.assertTrue(agent.episode.text.endswith("withheld; 2 model calls"))

    def test_unreadable_critique_is_an_objection(self):
        self.assertEqual(ca.parse_critique("NO ISSUES"), [])
        self.assertEqual([f.detail for f in ca.parse_critique("- a\n- b")], ["a", "b"])
        self.assertEqual(len(ca.parse_critique("Looks fine to me overall.")), 1)

    def test_reflection_keeps_a_gated_lesson_and_refuses_a_critic_one(self):
        agent, *_ = session(self.folder, [UNCITED, sc.GROUNDED], ["NO ISSUES"])
        reflector = scripted(AIMessage(content="", tool_calls=[
            {"name": "keep_lesson", "id": "k1", "type": "tool_call",
             "args": {"rule": "uncited-claim", "failed": 1, "fixed": 2,
                      "note": "The $40 bag fee sentence had no citation."}},
            {"name": "keep_lesson", "id": "k2", "type": "tool_call",
             "args": {"rule": "critic", "failed": 1, "fixed": 2, "note": "Bag fee."}}]))
        outcomes = agent.reflect(reflector)
        self.assertEqual(outcomes[0], "Kept as [lesson-1].")
        self.assertIn("must be one of", outcomes[1])
        self.assertIn("(check: uncited-claim)", reflector.seen[0][-1].content)

    def test_no_reflection_call_without_a_check_failure(self):
        agent, *_ = session(self.folder, [MISGROUNDED, sc.GROUNDED], ["- wrong case", "NO ISSUES"])
        self.assertEqual(agent.reflect(scripted()), [])

    def test_next_session_is_shown_lessons_as_a_tool_result(self):
        sc.LessonStore(self.folder, "flight-agent").add_lesson("uncited-claim", "Note.", "ev", "s0")
        agent, answer, model, critic = session(self.folder, [sc.GROUNDED], ["NO ISSUES"],
                                               day="2026-09-22")
        first = model.seen[0]
        self.assertEqual([m.content for m in first if isinstance(m, SystemMessage)], [ca.SYSTEM_PROMPT])
        notes = [m for m in first if isinstance(m, ToolMessage) and m.name == "recall_lessons"]
        self.assertIn("[lesson-1] uncited-claim: Note.", notes[0].content)

    def test_model_is_shown_the_contract_schemas(self):
        agent, *_ = session(self.folder, [sc.GROUNDED], ["NO ISSUES"])
        tools = {t.name: t for t in agent.agent.nodes["tools"].bound.tools_by_name.values()}
        self.assertEqual(set(tools), {"look_up_flight", "calculate", "search_policies",
                                      "recall_lessons", "review_answer"})
        for contract in ca.POLICY_CONTRACTS:
            self.assertEqual(tools[contract.name].args_schema, contract.input_schema)

    def test_review_turn_is_a_valid_anthropic_request(self):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            self.skipTest("langchain-anthropic is not installed")
        agent, *_ = session(self.folder, [UNCITED, sc.GROUNDED], ["NO ISSUES"])
        payload = ChatAnthropic(model="placeholder", api_key="unused")._get_request_payload(
            agent.messages[:-1])
        roles = [m["role"] for m in payload["messages"]]
        self.assertTrue(all(a != b for a, b in zip(roles, roles[1:])))
        turn = payload["messages"][-2]["content"]
        self.assertEqual([b["type"] for b in turn], ["text", "tool_use"])
        self.assertEqual(payload["messages"][-1]["content"][0]["type"], "tool_result")


if __name__ == "__main__":
    unittest.main()
