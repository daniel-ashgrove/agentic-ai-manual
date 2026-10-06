"""Tests the three composition decisions: the order the components run in, what
each of them may count as evidence, and what a run leaves behind.

Run: python -m unittest -v test_combined
"""

import logging
import tempfile
import unittest

import combined as c
import guardrails as g
from agent_memory import MemorySession, recall_report
from orchestration import Scripted, Worker, delegate, scripted_runner
from policy_retrieval import QUESTION
from self_critique import ScriptedModel, lesson_report

logging.getLogger("contracts").setLevel(logging.CRITICAL)

REFUND = [[("search_policies", {"query": "cancel within 24 hours refund"})],
          "Cancelling within 24 hours of departure means the fare becomes travel credit "
          "minus a $50 cancellation fee, valid for 12 months [refunds#2]."]
RULE = ("Cancelling 20 hours out is within 24 hours, so the fare becomes travel credit "
        "minus a $50 cancellation fee, valid for 12 months [refunds#2].")
UNCITED = RULE + " Your checked-bag fee is refunded whatever the timing."
CITED = UNCITED[:-1] + " [refunds#3]."
ASK = "What do I get back if I cancel 20 hours out?"
FROM_MEMORY = "The credit on BK-9910 comes to the $190 you were quoted."

MEMORY_Q = "Put through the credit we talked about on BK-9910."
MEMORY_ANSWER = ("Cancelling within 24 hours of departure means the fare becomes travel "
                 "credit minus a $50 cancellation fee [refunds#2], and the credit on "
                 "BK-9910 comes to the $190 you were quoted.")
CREDIT = ("request_credit", {"booking": "BK-9910", "amount_usd": 190,
                             "reason": "the credit the traveller was quoted [refunds#2]"})
SERVICE = [[CREDIT], "Recorded a request for $190 of travel credit on BK-9910."]
PROFILE = [[("recall_memory", {})],
           "The traveller was told support would give $190 back on BK-9910."]
QUOTED_FACT = {"key": "credit-quoted",
               "fact": "Was told support would give $190 back on BK-9910.",
               "quote": "Support told me I'd get $190 back on BK-9910"}


class Fixture(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stores = c.Stores(self.tmp.name, "T-41")
        self.ledger = g.Ledger()

    def run_one(self, question, coordinator, workers, critic=(), facts=(), notes=(),
                staff=None, composition=c.Composition(), limits=g.Limits(), day="2026-06-02"):
        return c.handle(
            question, "T-41", day, Scripted(*coordinator), ScriptedModel(list(critic)),
            lambda built: scripted_runner({name: Scripted(*replies)
                                           for name, replies in workers.items()}),
            self.ledger, self.stores, facts, notes, limits,
            staff or c.WORKERS, composition)

    def remember(self, day="2026-06-01"):
        """A fact the traveller stated in an earlier session, through Chapter 7's gate."""
        session = MemorySession(self.stores.memory(), day)
        session.hear("Support told me I'd get $190 back on BK-9910 if I cancel it.")
        c.ToolRegistry(c.make_memory_contracts(session)).execute("remember", QUOTED_FACT)

    def deliver(self, **kwargs):
        return self.run_one(ASK, [[delegate("policy", ASK)], RULE], {"policy": REFUND},
                            critic=[("critique", [])], **kwargs)


class Order(Fixture):

    def test_the_critique_runs_before_the_decision(self):
        run, composed = self.run_one(
            ASK, [[delegate("policy", ASK)], UNCITED], {"policy": REFUND},
            critic=[("revise", CITED), ("critique", [])])
        self.assertEqual([a.passes for a in composed.refinement.attempts], [False, True])
        self.assertTrue(composed.outcome.delivered)
        self.assertIsNone(composed.outcome.halted)

    def test_with_nothing_reserved_the_same_draft_is_escalated(self):
        run, composed = self.run_one(
            ASK, [[delegate("policy", ASK)], UNCITED], {"policy": REFUND},
            composition=c.Composition(critique_reserve=0))
        self.assertEqual(composed.outcome.halted, "unsupported-answer")
        self.assertIn("no citation", composed.outcome.why)

    def test_the_reserve_is_held_back_from_the_loop_not_taken_from_what_is_left(self):
        run, composed = self.run_one(
            ASK, [[delegate("policy", ASK)], UNCITED], {"policy": REFUND},
            critic=[("revise", CITED), ("critique", [])],
            limits=g.Limits(model_calls=9), composition=c.Composition(critique_reserve=5))
        self.assertTrue(composed.outcome.delivered)      # the loop spent 4 of 9
        self.assertEqual(composed.loop_calls, 4)
        self.assertEqual(run.calls, 6)                   # plus a revision and a review

    def test_a_delegation_is_refused_once_the_loop_has_spent_its_share(self):
        run, composed = self.run_one(
            ASK, [[delegate("policy", ASK)], [delegate("policy", "again, please")], RULE],
            {"policy": REFUND + REFUND}, critic=[("critique", [])],
            limits=g.Limits(model_calls=8), composition=c.Composition(critique_reserve=5))
        self.assertTrue(any("the loop's share" in line for line in run.log))
        self.assertTrue(composed.outcome.delivered)

    def test_the_coordinator_is_shown_memory_and_lessons_before_it_plans(self):
        seen = []

        def coordinator(question, transcript):
            seen.append(list(transcript))
            return RULE

        c.handle(ASK, "T-41", "2026-06-02", coordinator, ScriptedModel([]),
                 lambda built: scripted_runner({}), self.ledger, self.stores,
                 composition=c.Composition(critique_reserve=0))
        self.assertEqual(seen[0][0], recall_report(self.stores.memory()))
        self.assertEqual(seen[0][1], lesson_report(self.stores.lessons()))


class Admit(Fixture):

    def test_the_recall_report_is_in_the_answer_evidence_and_not_in_the_pool(self):
        self.remember()
        run, composed = self.run_one(
            MEMORY_Q, [[delegate("policy", ASK)], [delegate("service", "Request $190.")],
                       MEMORY_ANSWER],
            {"policy": REFUND, "service": SERVICE}, critic=[("critique", [])])
        pooled = c.pooled_evidence(composed.outcome.result.reports)
        self.assertNotIn(composed.recalled, pooled)
        self.assertIn(composed.recalled,
                      c.answer_evidence(composed.outcome.result, composed.recalled))
        self.assertEqual(composed.refinement.attempts[0].checks, ())   # the answer passes

    def test_an_amount_only_memory_knows_does_not_pass_the_spending_rule(self):
        self.remember()
        run, composed = self.run_one(
            MEMORY_Q, [[delegate("policy", ASK)], [delegate("service", "Request $190.")],
                       MEMORY_ANSWER],
            {"policy": REFUND, "service": SERVICE}, critic=[("critique", [])])
        self.assertEqual(composed.outcome.halted, "untraceable-amount")
        self.assertIn("$190 on BK-9910", composed.outcome.why)

    def test_the_same_amount_passes_once_a_worker_holds_the_memory_tool(self):
        self.remember()
        run, composed = self.run_one(
            MEMORY_Q,
            [[delegate("profile", "What were we told about BK-9910?")],
             [delegate("policy", ASK)], [delegate("service", "Request $190.")],
             MEMORY_ANSWER],
            {"profile": PROFILE, "policy": REFUND, "service": SERVICE},
            critic=[("critique", [])], staff=dict(c.WORKERS, profile=c.PROFILE))
        self.assertIn(composed.recalled,
                      c.pooled_evidence(composed.outcome.result.reports))
        self.assertTrue(composed.outcome.delivered)
        self.assertEqual([(a.booking, a.amount_usd) for a in composed.outcome.held],
                         [("BK-9910", 190)])

    def test_chapter_11s_rule_is_the_one_still_running(self):
        self.assertEqual([name for name, _ in g.HALT_CONDITIONS],
                         ["budget-spent", "repeated-failure", "untraceable-amount",
                          "unsupported-answer"])
        self.assertIs(dict(g.HALT_CONDITIONS)["untraceable-amount"],
                      g.untraceable_amount)

    def test_the_profile_workers_tool_is_bound_to_this_traveller(self):
        self.remember()
        run, composed = self.run_one(
            MEMORY_Q, [[delegate("profile", "What were we told?")], FROM_MEMORY],
            {"profile": PROFILE}, critic=[("critique", [])],
            staff=dict(c.WORKERS, profile=c.PROFILE))
        report = composed.outcome.result.reports[0]
        self.assertIn("BK-9910", report.evidence[0])
        self.assertEqual(c.PROFILE.contracts, ())      # the template holds no tool


class Keep(Fixture):

    def test_a_delivered_run_writes_one_episode_to_each_store(self):
        run, composed = self.deliver()
        self.assertEqual(len(self.stores.memory().episodes()), 1)
        self.assertEqual(len(self.stores.lessons().episodes()), 1)
        self.assertEqual(len(composed.written), 2)

    def test_a_halted_run_writes_nothing_at_all(self):
        run, composed = self.run_one(
            ASK, [[delegate("policy", ASK)], UNCITED], {"policy": REFUND},
            facts=(QUOTED_FACT,), composition=c.Composition(critique_reserve=0))
        self.assertFalse(composed.outcome.delivered)
        self.assertEqual(composed.written, ())
        self.assertEqual(self.stores.memory().records, [])
        self.assertEqual(self.stores.lessons().records, [])

    def test_the_write_gate_still_refuses_a_fact_the_traveller_did_not_state(self):
        run, composed = self.deliver(facts=(QUOTED_FACT,))
        self.assertTrue(any("REFUSED" in line for line in composed.written))
        self.assertEqual(self.stores.memory().facts(), [])

    def test_the_lesson_gate_still_refuses_a_lesson_with_no_failed_check(self):
        run, composed = self.deliver(notes=({"rule": "uncited-claim", "failed": 1,
                                             "fixed": 2, "note": "nothing failed."},))
        self.assertTrue(any("keep_lesson" in line and "REFUSED" in line
                            for line in composed.written))
        self.assertEqual(self.stores.lessons().lessons(), [])

    def test_a_lesson_is_kept_when_a_check_failed_and_a_later_attempt_passed(self):
        run, composed = self.run_one(
            ASK, [[delegate("policy", ASK)], UNCITED], {"policy": REFUND},
            critic=[("revise", CITED), ("critique", [])],
            notes=({"rule": "uncited-claim", "failed": 1, "fixed": 2,
                    "note": "The checked-bag sentence carried no citation; the "
                            "revision cited the refund passage."},))
        self.assertEqual(len(self.stores.lessons().lessons()), 1)

    def test_the_next_session_is_shown_what_the_last_one_kept(self):
        self.deliver()
        report = recall_report(self.stores.memory())
        self.assertIn("Recent sessions", report)
        self.assertIn("2026-06-02", report)


class TheWholeSystem(Fixture):

    def test_the_service_worker_is_added_and_holds_only_the_credit_tool(self):
        run, composed = self.deliver()
        self.assertEqual(c.service_worker(c.make_credit_contract(run, self.ledger))
                         .contracts[0].name, "request_credit")

    def test_every_gate_the_components_brought_is_still_in_the_path(self):
        run, composed = self.run_one(
            ASK, [[delegate("policy", ASK)], [delegate("policy", ASK)], RULE],
            {"policy": REFUND + REFUND}, critic=[("critique", [])])
        self.assertEqual(len(composed.outcome.result.reports), 1)   # Chapter 9's Gate 2


if __name__ == "__main__":
    unittest.main(verbosity=2)
