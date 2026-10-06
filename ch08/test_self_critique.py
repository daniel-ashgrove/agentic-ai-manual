"""Tests for self_critique.py. Standard library only; no model, no network.

Run: python -m unittest -v test_self_critique"""

import json
import tempfile
import unittest

import self_critique as sc
from agent_memory import MemoryStore, recall_report
from policy_retrieval import REPORT_HEADER
from tool_contracts import ToolRegistry

EVIDENCE = sc.gather_evidence()
UNCITED = sc.SESSIONS["2026-09-21"][0][1]
MISGROUNDED = sc.SESSIONS["2026-09-22"][0][1]
NO_FEE = sc.GROUNDED.replace(", minus a $50 cancellation fee", "")


def check(answer):
    return sc.check_findings(answer, sc.QUESTION, EVIDENCE)


class Counter:
    """A critic and a reviser that count their calls and reply from lists."""

    def __init__(self, critiques=(), revisions=()):
        self.critiques, self.revisions = list(critiques), list(revisions)
        self.critic_calls, self.revise_calls, self.findings_seen = 0, 0, []

    def critique(self, answer):
        self.critic_calls += 1
        return [sc.Finding("critic", "critic", t) for t in self.critiques.pop(0)]

    def revise(self, answer, findings):
        self.revise_calls += 1
        self.findings_seen.append(findings)
        return self.revisions.pop(0)

    def run(self, draft, max_revisions=2):
        return sc.refine(draft, check, self.critique, self.revise, max_revisions)


class CheckRules(unittest.TestCase):
    def test_every_chapter_6_problem_maps_to_a_rule(self):
        stale = "The first checked bag costs $35 [baggage-2025#1], so the total is $210."
        rules = {f.rule for f in check(stale) + check(UNCITED)}
        self.assertEqual(rules, {"unretrieved-citation", "unsourced-number", "uncited-claim"})
        forum = "[forum-4471#1] T (community post, not airline policy, posted 2026-05-10)\nx"
        found = sc.check_findings("Bags are free [forum-4471#1].", sc.QUESTION,
                                  [REPORT_HEADER + "\n\n" + forum])
        self.assertIn("not-current", {f.rule for f in found})

    def test_an_unknown_problem_fails_loudly(self):
        with self.assertRaises(ValueError):
            sc._rule_of("something the check never says")

    def test_grounded_answer_passes(self):
        self.assertEqual(check(sc.GROUNDED), [])


class Loop(unittest.TestCase):
    def test_critic_is_not_asked_while_checks_fail(self):
        c = Counter(critiques=[[]], revisions=[sc.GROUNDED])
        result = c.run(UNCITED)
        self.assertEqual((c.critic_calls, c.revise_calls, result.model_calls), (1, 1, 3))
        self.assertTrue(all(f.source == "check" for f in c.findings_seen[0]))
        self.assertEqual(result.delivered.number, 2)

    def test_critic_objection_triggers_a_revision(self):
        c = Counter(critiques=[["refunds#1 does not apply."], []], revisions=[sc.GROUNDED])
        result = c.run(MISGROUNDED)
        self.assertEqual(result.model_calls, 4)
        self.assertEqual(c.findings_seen[0][0].source, "critic")
        self.assertEqual(result.delivered.answer, sc.GROUNDED)

    def test_a_revision_that_fails_the_checks_is_not_delivered(self):
        c = Counter(critiques=[["wrong advice"]], revisions=[UNCITED])
        result = c.run(sc.GROUNDED, max_revisions=1)
        self.assertEqual(result.delivered.number, 1)
        self.assertFalse(result.attempts[-1].passes)
        self.assertEqual(result.stopped, "revision budget spent")

    def test_nothing_passes_means_withheld(self):
        c = Counter(revisions=[UNCITED, UNCITED])
        result = c.run(UNCITED)
        self.assertIsNone(result.delivered)
        self.assertEqual((len(result.attempts), result.model_calls), (3, 3))
        self.assertIn("withheld", result.stopped)

    def test_no_budget_means_no_critic_call(self):
        c = Counter()
        result = c.run(sc.GROUNDED, max_revisions=0)
        self.assertEqual((c.critic_calls, result.model_calls, result.delivered.number), (0, 1, 1))

    def test_last_revision_passing_is_delivered_without_a_critique(self):
        c = Counter(revisions=[sc.GROUNDED])
        result = c.run(UNCITED, max_revisions=1)
        self.assertEqual((c.critic_calls, result.model_calls), (0, 2))
        self.assertEqual(result.stopped, "passed the checks; no revision left to spend")

    def test_objection_is_recorded_on_the_attempt(self):
        c = Counter(critiques=[["x"], []], revisions=[sc.GROUNDED])
        result = c.run(sc.GROUNDED)
        self.assertEqual(result.attempts[0].critique[0].detail, "x")
        self.assertEqual(result.attempts[1].critique, ())

    def test_review_report_labels_who_found_what(self):
        report = sc.review_report([sc.Finding("check", "uncited-claim", "a\n b"),
                                   sc.Finding("critic", "critic", "c")])
        self.assertTrue(report.startswith(sc.REVIEW_HEADER))
        self.assertIn("- (check: uncited-claim) a b", report)
        self.assertIn("- (reviewer) c", report)

    def test_scripted_model_fails_loudly_out_of_step(self):
        with self.assertRaises(AssertionError):
            sc.ScriptedModel([("draft", "x")])("critique")


class Lessons(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.store = sc.LessonStore(self.folder, "flight-agent")
        self.result = sc.run_session(sc.SESSIONS["2026-09-21"], EVIDENCE)
        self.episode = sc.record_attempts(self.store, self.result, "s1")
        self.registry = ToolRegistry([sc.make_lesson_contract(
            self.store, self.result, "s1", self.episode.memory_id)])

    def keep(self, **changes):
        args = {"rule": "uncited-claim", "failed": 1, "fixed": 2,
                "note": "The $40 bag fee sentence had no citation."}
        return self.registry.execute("keep_lesson", {**args, **changes})

    def test_episode_holds_rule_names_and_counts_only(self):
        self.assertEqual(self.episode.text, "attempt-1 failed uncited-claim, unsourced-number; "
                                            "attempt-2 passed; delivered attempt-2; 3 model calls")
        self.assertNotIn("bag", self.episode.text)

    def test_valid_lesson_is_kept_with_its_evidence(self):
        self.assertTrue(self.keep().ok)
        lesson = self.store.lessons()[0]
        self.assertEqual((lesson.memory_id, lesson.kind, lesson.key), ("lesson-1", "lesson", "uncited-claim"))
        self.assertEqual(lesson.quote, "episode-1: attempt-1 failed uncited-claim; attempt-2 passed")

    def test_critic_is_not_a_rule(self):
        outcome = self.keep(rule="critic")
        self.assertEqual(outcome.error_kind, "invalid_arguments")

    def test_failed_attempt_must_have_failed_that_rule(self):
        outcome = self.keep(failed=2, fixed=2)
        self.assertIn("did not fail the 'uncited-claim' check", outcome.content)
        self.assertEqual(self.store.lessons(), [])

    def test_fixed_attempt_must_be_later_and_passing(self):
        self.assertIn("is not a later attempt", self.keep(fixed=1).content)
        self.assertIn("is not a later attempt", self.keep(fixed=7).content)

    def test_rule_words_are_refused(self):
        for note in ["Always cite the bag fee.", "Never state the $40 fee uncited.",
                     "Every bag fee needs a citation.", "Don\u2019t leave the bag fee uncited."]:
            self.assertIn("states a rule", self.keep(note=note).content, note)

    def test_note_must_share_a_word_and_its_numbers(self):
        self.assertIn("shares no words", self.keep(note="Something else entirely.").content)
        self.assertIn("states 60", self.keep(note="The bag fee was $60.").content)

    def test_one_current_lesson_per_rule(self):
        self.keep()
        self.keep(note="The bag fee sentence had no citation at all.")
        self.assertEqual([l.memory_id for l in self.store.lessons()], ["lesson-2"])
        self.assertEqual(self.store.records[1].superseded_by, "lesson-2")

    def test_lessons_survive_a_new_store_object_and_ids_continue(self):
        self.keep()
        again = sc.LessonStore(self.folder, "flight-agent")
        self.assertEqual(len(again.lessons()), 1)
        self.assertEqual(again.add_lesson("not-current", "x", "e", "s2").memory_id, "lesson-2")
        saved = json.loads(again.path.read_text())["counters"]
        self.assertEqual(saved, {"fact": 0, "episode": 1, "lesson": 2})

    def test_forget_all_then_new_lesson_starts_at_one(self):
        self.keep()
        self.store.forget_all()
        self.assertEqual(self.store.add_lesson("not-current", "x", "e", "s").memory_id, "lesson-1")

    def test_chapter_7_recall_never_shows_lessons(self):
        self.keep()
        self.assertEqual(recall_report(MemoryStore(self.folder, "flight-agent")).count("lesson"), 0)

    def test_lesson_report_labels_budgets_and_cleans(self):
        self.assertEqual(sc.lesson_report(self.store), sc.NO_LESSONS)
        for i in range(7):
            self.store.add_lesson(f"rule-{i}", f"note [x] {i}\nline", "ev", "s")
        report = sc.lesson_report(self.store)
        self.assertTrue(report.startswith(sc.LESSON_HEADER))
        self.assertEqual(report.count("\n[lesson-"), 5)
        self.assertIn("[lesson-7] rule-6: note (x) 6 line (evidence: ev)", report)

    def test_attempt_history_shows_each_attempt_and_finding(self):
        history = sc.attempt_history(self.result)
        self.assertIn("Attempt 1: Flight B costs $175", history)
        self.assertIn("(check: uncited-claim) makes a claim with no citation", history)
        self.assertTrue(history.endswith("passed every check"))


if __name__ == "__main__":
    unittest.main()
