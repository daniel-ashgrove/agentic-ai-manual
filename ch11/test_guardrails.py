"""Tests for the guardrail layer: the limits, the acting tool, the halt conditions,
the escalation record and the demonstration's printed figures.

Run: python -m unittest -v test_guardrails
"""

import io
import unittest
from contextlib import redirect_stdout

import guardrails as g
from orchestration import Report, Scripted, WORKERS, orchestrate, scripted_runner
from tool_contracts import ToolRegistry


def a_run(**kwargs) -> g.Run:
    return g.Run(kwargs.pop("traveller", "T-1"), kwargs.pop("question", "cancel BK-1"),
                 "2026-06-02", kwargs.pop("limits", g.Limits()))


def registry_for(run: g.Run, ledger: g.Ledger) -> ToolRegistry:
    return ToolRegistry([g.make_credit_contract(run, ledger)])


def result_with(*reports, problems=(), draft="an answer") -> g.Orchestration:
    return g.Orchestration(draft, not problems, draft, tuple(problems), reports, 1)


def report(worker="fares", answer="Flight B costs $175.", evidence=("price_usd: 175",),
           problems=()):
    return Report(worker, "a task", answer, tuple(evidence), tuple(problems), 1)


class TheActingTool(unittest.TestCase):

    def test_records_a_request_and_issues_nothing(self):
        run, ledger = a_run(), g.Ledger()
        outcome = registry_for(run, ledger).execute(
            "request_credit", {"booking": "BK-1", "amount_usd": 125,
                               "reason": "fare minus the fee [refunds#2]"})
        self.assertTrue(outcome.ok)
        self.assertEqual([(a.booking, a.amount_usd) for a in run.held], [("BK-1", 125)])
        self.assertEqual(ledger.applied, [])          # nothing has moved
        self.assertIn("No credit has been issued", outcome.content)

    def test_the_echo_names_the_amount_so_a_report_can_be_checked(self):
        run = a_run()
        outcome = registry_for(run, g.Ledger()).execute(
            "request_credit", {"booking": "BK-1", "amount_usd": 125, "reason": "x" * 10})
        self.assertIn("$125", outcome.content)
        self.assertIn("BK-1", outcome.content)

    def test_an_amount_over_the_request_limit_is_refused_at_gate_2(self):
        run, ledger = a_run(limits=g.Limits(credit_usd=300)), g.Ledger()
        outcome = registry_for(run, ledger).execute(
            "request_credit", {"booking": "BK-1", "amount_usd": 900, "reason": "x" * 10})
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.error_kind, "rejected")
        self.assertEqual(run.held, [])
        self.assertTrue(any("over the $300 request limit" in line for line in run.log))

    def test_the_same_booking_is_not_requested_twice_in_one_run(self):
        run, ledger, registry = a_run(), g.Ledger(), None
        registry = registry_for(run, ledger)
        args = {"booking": "BK-1", "amount_usd": 125, "reason": "x" * 10}
        self.assertTrue(registry.execute("request_credit", args).ok)
        second = registry.execute("request_credit", args)
        self.assertFalse(second.ok)
        self.assertIn("already recorded", second.content)
        self.assertEqual(len(run.held), 1)

    def test_the_daily_rate_limit_counts_issued_credit_not_requests(self):
        ledger = g.Ledger()
        first = a_run(limits=g.Limits(credits_per_day=1))
        self.assertTrue(registry_for(first, ledger).execute(
            "request_credit", {"booking": "BK-1", "amount_usd": 50,
                               "reason": "x" * 10}).ok)
        later = a_run(limits=g.Limits(credits_per_day=1))
        self.assertTrue(registry_for(later, ledger).execute(       # still only requested
            "request_credit", {"booking": "BK-2", "amount_usd": 50,
                               "reason": "x" * 10}).ok)
        g.review(first.held[0], "dana", ledger, approve=True)      # now one is issued
        third = a_run(limits=g.Limits(credits_per_day=1))
        outcome = registry_for(third, ledger).execute(
            "request_credit", {"booking": "BK-3", "amount_usd": 50, "reason": "x" * 10})
        self.assertFalse(outcome.ok)
        self.assertIn("a day allows", outcome.content)

    def test_the_schema_still_rejects_a_malformed_call_at_gate_1(self):
        outcome = registry_for(a_run(), g.Ledger()).execute(
            "request_credit", {"booking": "BK-1", "amount_usd": "one hundred",
                               "reason": "x" * 10})
        self.assertEqual(outcome.error_kind, "invalid_arguments")

    def test_the_service_worker_holds_only_the_one_tool(self):
        worker = g.service_worker(g.make_credit_contract(a_run(), g.Ledger()))
        self.assertEqual([c.name for c in worker.contracts], ["request_credit"])


class TheLedger(unittest.TestCase):

    def test_only_review_issues_credit(self):
        run, ledger = a_run(), g.Ledger()
        registry_for(run, ledger).execute(
            "request_credit", {"booking": "BK-1", "amount_usd": 125, "reason": "x" * 10})
        self.assertEqual(ledger.count("T-1", "2026-06-02"), 0)
        line = g.review(run.held[0], "dana", ledger, approve=True)
        self.assertEqual(ledger.count("T-1", "2026-06-02"), 1)
        self.assertIn("approved by dana", line)

    def test_a_refusal_is_recorded_and_issues_nothing(self):
        run, ledger = a_run(), g.Ledger()
        registry_for(run, ledger).execute(
            "request_credit", {"booking": "BK-1", "amount_usd": 125, "reason": "x" * 10})
        line = g.review(run.held[0], "dana", ledger, approve=False, why="wrong flight")
        self.assertEqual((len(ledger.applied), len(ledger.refused)), (0, 1))
        self.assertIn("wrong flight", line)


class HaltConditions(unittest.TestCase):

    def test_an_amount_from_a_tool_result_passes(self):
        run = a_run()
        run.held.append(g.Action("T-1", "BK-1", 125, "because", "2026-06-02"))
        result = result_with(report(evidence=("125.0",)))
        self.assertIsNone(g.untraceable_amount(run, result))

    def test_an_amount_in_no_tool_result_halts(self):
        run = a_run()
        run.held.append(g.Action("T-1", "BK-1", 85, "because", "2026-06-02"))
        why = g.untraceable_amount(run, result_with(report(evidence=("125.0",))))
        self.assertIn("$85", why)

    def test_the_requests_own_echo_is_not_evidence_for_it(self):
        run = a_run()
        run.held.append(g.Action("T-1", "BK-1", 125, "because", "2026-06-02"))
        echo = report("service", "Recorded $125 on BK-1.", ("Recorded $125 on BK-1.",))
        self.assertIsNotNone(g.untraceable_amount(run, result_with(echo)))

    def test_a_number_the_traveller_wrote_counts_as_seen(self):
        run = a_run(question="refund my $125, booking BK-1")
        run.held.append(g.Action("T-1", "BK-1", 125, "because", "2026-06-02"))
        self.assertIsNone(g.untraceable_amount(run, result_with()))

    def test_a_withheld_report_contributes_no_evidence(self):
        run = a_run()
        run.held.append(g.Action("T-1", "BK-1", 125, "because", "2026-06-02"))
        failed = report(evidence=("125.0",), problems=("states 125 with no source",))
        self.assertIsNotNone(g.untraceable_amount(run, result_with(failed)))

    def test_a_failed_final_check_escalates_instead_of_only_withholding(self):
        why = g.unsupported_answer(a_run(), result_with(problems=("cites [x#1]",)))
        self.assertIn("cites [x#1]", why)

    def test_two_failed_reports_in_a_row_halt_and_two_apart_do_not(self):
        bad, good = report(problems=("no",)), report()
        self.assertIsNotNone(g.repeated_failure(a_run(), result_with(bad, bad)))
        self.assertIsNone(g.repeated_failure(a_run(), result_with(bad, good, bad)))

    def test_the_budget_condition_reads_the_runs_own_count(self):
        run = a_run(limits=g.Limits(model_calls=6))
        run.calls = 6
        self.assertIn("6 model calls", g.budget_spent(run, result_with()))

    def test_the_first_condition_to_fire_is_the_one_recorded(self):
        run = a_run(limits=g.Limits(model_calls=2))
        run.calls, _ = 5, run.held.append(
            g.Action("T-1", "BK-1", 85, "because", "2026-06-02"))
        outcome = g.decide(run, result_with(problems=("cites [x#1]",)))
        self.assertEqual(outcome.halted, "budget-spent")


class WhatLeavesTheSystem(unittest.TestCase):

    def test_a_halted_run_tells_the_traveller_nothing_moved(self):
        run = a_run(limits=g.Limits(model_calls=1))
        run.calls = 2
        outcome = g.decide(run, result_with())
        self.assertFalse(outcome.delivered)
        self.assertIn("nothing has been charged or credited", outcome.answer)

    def test_a_delivered_answer_states_the_amount_and_the_wait(self):
        run = a_run()
        run.held.append(g.Action("T-1", "BK-4471", 125, "because", "2026-06-02"))
        outcome = g.decide(run, result_with(report(evidence=("125.0",))))
        self.assertTrue(outcome.delivered)
        self.assertIn("$125 of travel credit on BK-4471", outcome.answer)
        self.assertIn("one business day", outcome.answer)

    def test_the_escalation_carries_the_draft_the_traveller_never_saw(self):
        run = a_run()
        run.held.append(g.Action("T-1", "BK-1", 85, "fare minus fees", "2026-06-02"))
        outcome = g.decide(run, result_with(report(), draft="The credit is $85."))
        packet = g.escalation(run, outcome)
        self.assertIn("untraceable-amount", packet)
        self.assertIn("The credit is $85.", packet)
        self.assertIn("not issued", packet)

    def test_every_decision_reaches_the_log(self):
        run, ledger = a_run(), g.Ledger()
        registry_for(run, ledger).execute(
            "request_credit", {"booking": "BK-1", "amount_usd": 125, "reason": "x" * 10})
        g.decide(run, result_with(report(evidence=("125.0",))))
        self.assertEqual([line.split()[2] for line in run.log], ["held", "delivered"])


class TheWholeRun(unittest.TestCase):

    def run_one(self, index: int):
        title, traveller, question, coordinator, workers, limits = g.RUNS[index]
        return g.guarded(question, traveller, g.DAY, Scripted(*coordinator),
                         lambda built: scripted_runner(
                             {name: Scripted(*replies)
                              for name, replies in workers.items()}),
                         g.Ledger(), limits)

    def test_a_run_with_nothing_to_stop_is_delivered_unchanged(self):
        run, outcome = self.run_one(0)
        self.assertTrue(outcome.delivered)
        self.assertEqual((outcome.calls, len(outcome.held)), (7, 0))

    def test_a_good_request_is_held_and_the_run_still_answers(self):
        run, outcome = self.run_one(1)
        self.assertTrue(outcome.delivered)
        self.assertEqual([(a.booking, a.amount_usd) for a in outcome.held],
                         [("BK-4471", 125)])

    def test_an_amount_no_tool_produced_stops_the_same_question(self):
        run, outcome = self.run_one(2)
        self.assertEqual(outcome.halted, "untraceable-amount")
        self.assertIn("$85", outcome.why)

    def test_the_rule_stops_a_correct_amount_too(self):
        run, outcome = self.run_one(3)
        self.assertEqual(outcome.halted, "untraceable-amount")
        self.assertIn("$125", outcome.why)          # the right amount, and still stopped

    def test_the_budget_is_enforced_while_the_run_happens(self):
        run, outcome = self.run_one(5)
        self.assertEqual(outcome.halted, "budget-spent")
        self.assertTrue(any("no model calls left" in line for line in outcome.log))
        self.assertEqual(len(outcome.result.reports), 2)   # the third never ran

    def test_the_demonstration_prints_the_same_thing_twice(self):
        first, second = io.StringIO(), io.StringIO()
        with redirect_stdout(first):
            g.run_demo()
        with redirect_stdout(second):
            g.run_demo()
        self.assertEqual(first.getvalue(), second.getvalue())
        self.assertIn("Recorded: 3; issued after approval: 1", first.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
