"""Tests for eval_harness.py. Standard library only; no model, no network.

Run: python -m unittest -v test_eval_harness"""

import math
import tempfile
import unittest
from pathlib import Path

import eval_harness as eh
from self_critique import GROUNDED, LessonStore

CASE = {c.case_id: c for c in eh.CASES}
NO_FEE = GROUNDED.replace(", minus a $50 cancellation fee", "")


def scores(outcomes, calls=1):
    """A run from a string per case, one letter per trial: c correct, w wrong, h withheld."""
    names = {"c": "correct", "w": "wrong", "h": "withheld"}
    return [eh.Score(case_id, n, names[o], o != "h", (), calls)
            for case_id, trials in outcomes.items() for n, o in enumerate(trials, start=1)]


class References(unittest.TestCase):
    def test_bag_totals_follow_the_policy(self):
        self.assertEqual([eh.bag_total(f) for f in "ABC"], [210, 215, 280])
        self.assertEqual((eh.bag_total("B", 0), eh.bag_total("B", 2)), (175, 275))

    def test_the_cheapest_flight_depends_on_the_bag(self):
        self.assertEqual(eh.cheapest_arriving_before("18:00"), "A")
        self.assertEqual(eh.cheapest_arriving_before("18:00", bags=0), "B")

    def test_references_are_computed_not_typed(self):
        self.assertEqual(CASE["cheapest-with-bag"].must_state, ("flight A", "$210"))
        self.assertIn("$215", CASE["bag-and-cancel"].must_state)


class Grader(unittest.TestCase):
    def test_phrases_match_whole(self):
        self.assertTrue(eh._states("a $50 fee.", "$50"))
        self.assertFalse(eh._states("a $500 fee", "$50"))
        self.assertFalse(eh._states("the flight arrives", "flight a"))
        self.assertTrue(eh._states("Flight A is cheapest", "flight a"))

    def test_correct_answers_pass(self):
        for case_id, answer in [("bag-and-cancel", GROUNDED), ("cheapest-with-bag", eh.CHEAPEST),
                                ("bag-A", eh.BAG_A), ("cancel-early", eh.EARLY)]:
            self.assertEqual(eh.grade(CASE[case_id], answer), [], case_id)

    def test_wrong_answers_are_named(self):
        self.assertEqual(eh.grade(CASE["bag-and-cancel"], NO_FEE), ["does not state '$50'"])
        self.assertIn("states 'refunded in full'",
                      eh.grade(CASE["bag-and-cancel"], eh.MISGROUNDED))
        self.assertEqual(len(eh.grade(CASE["cheapest-with-bag"], eh.SPLIT)), 2)
        self.assertEqual(eh.grade(CASE["bag-A"], eh.BAG_A_WITH_FEE), ["states '$250'"])

    def test_withheld_is_its_own_outcome(self):
        self.assertEqual(eh.grade(CASE["bag-A"], None), ["withheld"])
        t = eh.score_trial(CASE["bag-A"], eh.Trial("bag-A", 1, None, (), 1))
        self.assertEqual((t.outcome, t.traceable), ("withheld", False))

    def test_traceable_is_not_correct(self):
        system = eh.scripted_system(eh.PLAIN, 2, shown=("lesson-2",))
        s = eh.score_trial(CASE["bag-A"], system(CASE["bag-A"], 1))
        self.assertEqual((s.traceable, s.outcome), (True, "wrong"))


class Runs(unittest.TestCase):
    def test_every_case_and_trial_in_order(self):
        run = eh.run_eval(eh.scripted_system(eh.PLAIN, 0), trials=2)
        self.assertEqual([(s.case_id, s.trial) for s in run][:3],
                         [("bag-and-cancel", 1), ("bag-and-cancel", 2), ("cheapest-with-bag", 1)])
        self.assertEqual(len(run), 8)

    def test_budget_zero_withholds_what_fails_the_check(self):
        s = eh.run_eval(eh.scripted_system(eh.PLAIN, 0))[0]
        self.assertEqual((s.outcome, s.model_calls), ("withheld", 1))

    def test_summary_counts_every_trial_and_some_trial(self):
        t = eh.summarize(scores({"x": "ccw", "y": "ccc", "z": "hww"}))
        self.assertEqual((t["correct"], t["wrong"], t["withheld"], t["delivered"]), (5, 3, 1, 8))
        self.assertEqual((t["every_trial_correct"], t["some_trial_correct"]), (1, 2))


class Compare(unittest.TestCase):
    def test_paired_counts_and_per_case_means(self):
        c = eh.compare(scores({"x": "wwc", "y": "www"}), scores({"x": "ccw", "y": "ccc"}, 2))
        self.assertEqual((c.fixed, c.broke, c.better_cases, c.worse_cases), (5, 1, 2, 0))
        self.assertAlmostEqual(c.mean_difference, (1 / 3 + 1) / 2)
        self.assertAlmostEqual(c.standard_error, 1 / 3)
        self.assertEqual(c.extra_calls, 6)

    def test_runs_that_do_not_pair_are_refused(self):
        with self.assertRaises(ValueError):
            eh.compare(scores({"x": "cc"}), scores({"x": "c"}))

    def test_one_case_has_no_standard_error(self):
        self.assertTrue(math.isnan(eh.compare(scores({"x": "c"}), scores({"x": "w"})).standard_error))


class LogAndDecisions(unittest.TestCase):
    def test_the_log_appends_and_survives_a_new_object(self):
        path = Path(tempfile.mkdtemp()) / "runs.jsonl"
        eh.ScoreLog(path).record("A", {"lessons": []}, scores({"x": "cw"}))
        eh.ScoreLog(path).record("B", {"lessons": ["l"]}, scores({"x": "cc"}))
        runs = eh.ScoreLog(path).runs()
        self.assertEqual([(r["run"], r["correct"]) for r in runs], [("A", 1), ("B", 2)])
        self.assertEqual(len(runs[0]["scores"]), 2)

    def test_lesson_decisions(self):
        with_it = scores({"x": "cw", "y": "cc"})
        self.assertEqual(eh.decide_lesson(with_it, scores({"x": "cc", "y": "cc"}))[0], "retire")
        self.assertEqual(eh.decide_lesson(with_it, scores({"x": "cw", "y": "cw"}))[0], "keep")
        self.assertEqual(eh.decide_lesson(with_it, scores({"x": "cc", "y": "cw"}))[0], "review")
        self.assertEqual(eh.decide_lesson(with_it, with_it)[0], "keep")

    def test_attempt_records_are_read_from_chapter_8_episodes(self):
        store = LessonStore(tempfile.mkdtemp(), "flight-agent")
        eh.run_eval(eh.scripted_system(eh.PLAIN, 2, store, "A"))
        eh.run_eval(eh.scripted_system(eh.PLAIN, 0, store, "Z"))
        self.assertEqual(eh.attempt_record(store, "A"),
                         {"revisions_per_answer": 0.5, "first_drafts_passing": 11})
        self.assertEqual(eh.attempt_record(store, "Z")["revisions_per_answer"], 0)


class Lessons(unittest.TestCase):
    def test_chapter_8_lessons_are_kept_through_its_gate(self):
        store = LessonStore(tempfile.mkdtemp(), "flight-agent")
        eh.keep_chapter_8_lessons(store)
        self.assertEqual([(l.memory_id, l.key) for l in store.lessons()],
                         [("lesson-1", "uncited-claim"), ("lesson-2", "unsourced-number")])

    def test_a_lesson_changes_only_its_drafts(self):
        plain = eh.run_eval(eh.scripted_system(eh.PLAIN, 2))
        shown = eh.run_eval(eh.scripted_system(eh.PLAIN, 2, shown=("lesson-2",)))
        changed = {s.case_id for s, t in zip(plain, shown) if s.outcome != t.outcome}
        self.assertEqual(changed, {"bag-A"})

    def test_the_demo_retires_lesson_2_and_keeps_lesson_1(self):
        runs = self._demo()
        self.assertEqual([r["config"]["lessons"] for r in runs if r["run"] == "C"],
                         [["lesson-1"]])

    def _demo(self):
        import contextlib
        import io
        import os
        here = os.getcwd()
        os.chdir(tempfile.mkdtemp())
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                eh.run_demo()
            return eh.ScoreLog(eh.LOG).runs()
        finally:
            os.chdir(here)


if __name__ == "__main__":
    unittest.main()
