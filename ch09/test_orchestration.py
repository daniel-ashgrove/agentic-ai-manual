"""Tests for orchestration.py: scope, briefs, reports, the delegation contract, the
merge check and the demonstration. Standard library only.

Run: python -m unittest -v test_orchestration"""

import contextlib
import io
import logging
import threading
import time
import unittest

import orchestration as o

logging.getLogger("tool_contracts").setLevel(logging.CRITICAL)
FARES, POLICY = o.WORKERS["fares"], o.WORKERS["policy"]
RUN1, RUN2, RUN3A, RUN3B = o.RUNS


def replay(run, **kwargs):
    title, question, coordinator, workers = run
    runner = o.scripted_runner({n: o.Scripted(*r) for n, r in workers.items()})
    return o.orchestrate(question, o.Scripted(*coordinator), o.WORKERS, runner, **kwargs)


def search_ids(worker, query):
    registry = o.ToolRegistry(list(worker.contracts))
    content = registry.execute("search_policies", {"query": query, "max_results": 5}).content
    return [line.split("]")[0][1:] for line in content.splitlines() if line.startswith("[")]


class Scope(unittest.TestCase):
    def test_each_worker_has_only_its_tools(self):
        self.assertEqual([c.name for c in FARES.contracts],
                         ["look_up_flight", "calculate", "search_policies"])
        self.assertEqual([c.name for c in POLICY.contracts], ["search_policies"])

    def test_slices_reach_only_their_documents(self):
        fares = search_ids(FARES, "cancel refund pet cat fee")
        policy = search_ids(POLICY, "checked bag fee carry-on")
        self.assertTrue(fares and all(i.startswith("baggage-2026") for i in fares))
        self.assertTrue(policy and all(i.split("#")[0] in ("refunds", "pets") for i in policy))

    def test_out_of_scope_call_is_refused_and_is_not_evidence(self):
        runner = o.scripted_runner({"policy": o.Scripted(o.lookups("B"), "done")})
        answer, evidence, calls = runner(POLICY, "Look up flight B for me, please.")
        self.assertEqual((evidence, calls), ([], 2))

    def test_description_says_what_the_slice_covers(self):
        self.assertIn("baggage policies", FARES.contracts[-1].description)
        self.assertIn("cancellation, refund and pet policies", POLICY.contracts[0].description)


class Briefs(unittest.TestCase):
    def test_delegate_schema_lists_the_workers(self):
        contract = o.make_delegate_contract(o.WORKERS, o.QUESTION, None, [])
        self.assertEqual(contract.input_schema["properties"]["worker"]["enum"],
                         ["fares", "policy"])
        self.assertIn("fares: Prices flights", contract.description)

    def test_unknown_worker_is_refused_at_gate_1(self):
        registry = o.ToolRegistry([o.make_delegate_contract(o.WORKERS, o.QUESTION, None, [])])
        outcome = registry.execute("delegate", {"worker": "pets", "task": "Price a cat, please."})
        self.assertEqual(outcome.error_kind, "invalid_arguments")

    def test_budget_is_enforced(self):
        reports = []
        runner = o.scripted_runner({"fares": o.Scripted(*o.FARES_B)})
        registry = o.ToolRegistry([o.make_delegate_contract(o.WORKERS, o.QUESTION, runner,
                                                            reports, max_delegations=1)])
        first = registry.execute("delegate", {"worker": "fares", "task": o.PRICE_B})
        second = registry.execute("delegate", {"worker": "policy", "task": o.CANCEL})
        self.assertTrue(first.ok)
        self.assertEqual(second.error_kind, "rejected")
        self.assertIn("no delegations left (the limit is 1)", second.content)
        self.assertEqual(len(reports), 1)

    def test_a_repeated_brief_is_refused(self):
        runner = o.scripted_runner({"fares": o.Scripted(*o.FARES_B)})
        registry = o.ToolRegistry([o.make_delegate_contract(o.WORKERS, o.QUESTION, runner, [])])
        registry.execute("delegate", {"worker": "fares", "task": o.PRICE_B})
        again = registry.execute("delegate", {"worker": "fares", "task": "  " + o.PRICE_B})
        self.assertIn("already had this exact task", again.content)

    def test_admission_is_atomic_when_calls_run_at_once(self):
        def slow(worker, task):
            time.sleep(0.2)
            return "", [], 1
        reports = []
        registry = o.ToolRegistry([o.make_delegate_contract(o.WORKERS, o.QUESTION, slow,
                                                            reports, max_delegations=1)])
        results = []
        threads = [threading.Thread(target=lambda t=t: results.append(registry.execute(
            "delegate", {"worker": "fares", "task": f"Task number {t}, please."})))
            for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual((len(reports), sum(r.ok for r in results)), (1, 1))

    def test_numbers_in_a_brief_are_not_evidence(self):
        result = replay(RUN2)
        smuggled = result.reports[0]
        self.assertIn("states 35 with no source", smuggled.problems[0])
        self.assertEqual(o.check_answer(smuggled.answer, smuggled.task,
                                        list(smuggled.evidence)), [])


class Reports(unittest.TestCase):
    def test_a_failed_report_is_withheld_from_the_coordinator(self):
        coordinator = o.Scripted(RUN2[2][0], RUN2[2][1], RUN2[2][2])
        seen = []
        original = coordinator.__call__

        def watch(question, transcript):
            seen.append(transcript)
            return original(question, transcript)
        runner = o.scripted_runner({n: o.Scripted(*r) for n, r in RUN2[3].items()})
        o.orchestrate(o.QUESTION, watch, o.WORKERS, runner)
        first_reports = " ".join(seen[1])
        self.assertIn("[report from fares withheld] It did not pass the checks: states 35 "
                      "with no source", first_reports)
        self.assertNotIn("$210", first_reports)

    def test_a_passing_report_keeps_its_citations(self):
        report = replay(RUN1).reports[1]
        self.assertEqual(o.for_coordinator(report), "[report from policy] " + report.answer)
        self.assertIn("[refunds#2]", report.answer)


class Merge(unittest.TestCase):
    def test_a_citation_one_worker_retrieved_is_valid_in_the_answer(self):
        result = replay(RUN1)
        self.assertTrue(result.delivered)
        self.assertEqual(result.answer, o.MERGED)

    def test_reports_are_never_evidence(self):
        coordinator = RUN2[2][:1] + ["Flight B costs $175, and with the $35 checked-bag fee "
                                     "the total is $210."]
        result = replay(("", o.QUESTION, coordinator, RUN2[3]))
        self.assertFalse(result.delivered)
        self.assertEqual(result.answer, o.WITHHELD)
        self.assertTrue(any(p.startswith("states 35 with no source") for p in result.problems))
        as_if_evidence = o.pooled_evidence(result.reports) + [r.answer for r in result.reports]
        self.assertEqual(o.check_answer(result.draft, o.QUESTION, as_if_evidence), [])

    def test_the_turn_limit_withholds(self):
        coordinator = [[o.delegate("fares", f"Look up flight {f}, please.")] for f in "ABC"]
        workers = {"fares": [o.lookups("A"), "Flight A costs $210.",
                             o.lookups("B"), "Flight B costs $175."]}
        result = replay(("", o.QUESTION, coordinator, workers), max_turns=2)
        self.assertFalse(result.delivered)
        self.assertEqual(result.problems, ("no answer within 2 coordinator turns",))
        self.assertEqual(result.coordinator_calls, 2)

    def test_model_calls_are_counted_at_both_levels(self):
        result = replay(RUN1)
        self.assertEqual((result.coordinator_calls, result.worker_calls), (2, 5))
        self.assertEqual([r.model_calls for r in replay(RUN2).reports], [3, 2, 3])


    def test_a_failed_run_contributes_no_evidence(self):
        result = replay(RUN2)
        self.assertEqual(len(o.pooled_evidence(result.reports)), 4)
        self.assertNotIn(result.reports[0].evidence[-1], o.pooled_evidence(result.reports))

    def test_known_limit_a_brief_number_can_enter_through_the_calculator(self):
        """Not fixed here: the check sees a tool's result, not where its input came from."""
        runner = o.scripted_runner({"fares": o.Scripted(
            o.lookups("B"), [("calculate", {"expression": "175 + 35"})],
            "Flight B with one checked bag comes to $210 in total.")})
        answer, evidence, calls = runner(FARES, "Price flight B; the bag fee is $35.")
        self.assertEqual(o.check_answer(answer, o.QUESTION, evidence), [])


class Demonstration(unittest.TestCase):
    def test_the_split_that_loses_a_condition_passes_and_is_wrong(self):
        wrong, right = replay(RUN3A), replay(RUN3B)
        self.assertTrue(wrong.delivered and right.delivered)
        self.assertIn("Flight B is the cheapest", wrong.answer)
        self.assertIn("Flight A is the cheapest", right.answer)
        self.assertEqual((wrong.coordinator_calls + wrong.worker_calls,
                          right.coordinator_calls + right.worker_calls), (8, 5))

    def test_the_scripted_model_fails_out_of_step(self):
        with self.assertRaises(AssertionError):
            o.Scripted()("anything")

    def test_the_demo_is_deterministic(self):
        outputs = []
        for _ in range(2):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                o.run_demo()
            outputs.append(buffer.getvalue())
        self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
