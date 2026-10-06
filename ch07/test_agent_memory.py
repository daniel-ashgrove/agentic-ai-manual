"""Edge cases behind the chapter's demonstration.

Run: python -m unittest -v test_agent_memory"""

import json
import os
import tempfile
import unittest

from agent_memory import (NO_MEMORY, CONTRACTS, MemorySession, MemoryStore,
                          MemoryStoreError, ToolRegistry, check_memory_write,
                          close_session, make_memory_contracts, recall_report)

SAID = "Wait, I always check a bag. Does that change it?"


def gate(turns, **args):
    session = MemorySession(None, "s")
    for t in turns:
        session.hear(t)
    base = {"key": "checked-bags", "fact": "Checks one bag on every trip.",
            "quote": "I always check a bag"}
    return check_memory_write(session, {**base, **args})


class Store(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()

    def store(self, user="t-1041"):
        return MemoryStore(self.folder, user)

    def test_user_ids_that_could_leave_the_folder_are_refused(self):
        for bad in ["../t-1041", "t/1041", "T-1041", "", "-t", "t 1041"]:
            with self.assertRaises(ValueError, msg=bad):
                MemoryStore(self.folder, bad)

    def test_a_new_store_object_reads_what_another_wrote(self):
        self.store().add_fact("checked-bags", "Checks one bag.", "I always check a bag", "s1")
        self.assertEqual([f.text for f in self.store().facts()], ["Checks one bag."])

    def test_a_fact_with_the_same_key_supersedes_and_the_old_one_is_kept(self):
        s = self.store()
        s.add_fact("checked-bags", "Checks one bag.", "q", "s1")
        s.add_fact("checked-bags", "Carry-on only.", "q", "s2")
        s.add_fact("seat", "Aisle seat.", "q", "s2")
        self.assertEqual([f.memory_id for f in s.facts()], ["fact-2", "fact-3"])
        self.assertEqual(s.records[0].superseded_by, "fact-2")

    def test_forget_removes_every_version_of_a_fact(self):
        s = self.store()
        s.add_fact("checked-bags", "Checks one bag.", "q", "s1")
        s.add_fact("checked-bags", "Carry-on only.", "q", "s2")
        s.add_episode("Asked: x", "s2")
        self.assertEqual(s.forget("fact-2"), 2)
        self.assertEqual([r.memory_id for r in self.store().records], ["episode-1"])
        self.assertEqual(s.forget("fact-9"), 0)

    def test_ids_are_never_reused_after_a_deletion(self):
        s = self.store()
        s.add_fact("a", "A.", "q", "s1")
        s.forget("fact-1")
        self.assertEqual(self.store().add_fact("b", "B.", "q", "s2").memory_id, "fact-2")

    def test_an_unreadable_file_fails_loudly_and_is_not_overwritten(self):
        path = os.path.join(self.folder, "t-1041.json")
        for content in ["{not json", json.dumps([{"memory_id": "fact-1"}]), "[]"]:
            with open(path, "w") as f:
                f.write(content)
            with self.assertRaises(MemoryStoreError):
                self.store()
            with open(path) as f:
                self.assertEqual(f.read(), content)

    def test_saving_leaves_no_temporary_files(self):
        self.store().add_episode("Asked: x", "s1")
        self.assertEqual(os.listdir(self.folder), ["t-1041.json"])

    def test_travellers_do_not_share_memory(self):
        self.store().add_fact("a", "A.", "q", "s1")
        self.assertEqual(recall_report(self.store("t-2207")), NO_MEMORY)


class WriteGate(unittest.TestCase):
    def test_a_quote_the_traveller_said_is_accepted(self):
        self.assertEqual(gate([SAID]), [])

    def test_quotes_match_despite_case_spacing_and_curly_apostrophes(self):
        self.assertEqual(gate(["I\u2019m going  CARRY-ON only from now on"],
                              fact="Travels carry-on only.",
                              quote="i'm going carry-on only"), [])

    def test_a_quote_nobody_said_is_refused(self):
        self.assertIn("does not appear", gate([SAID], quote="I prefer an aisle seat")[0])

    def test_a_quote_from_an_earlier_session_is_refused(self):
        self.assertIn("does not appear", gate([])[0])

    def test_a_short_quote_is_refused_even_if_said(self):
        self.assertIn("at least four words", gate(["Yes"], quote="Yes")[0])

    def test_a_fact_sharing_no_words_with_its_quote_is_refused(self):
        problems = gate(["Yes, flight A it is."], fact="Bag fees are waived.",
                        quote="Yes, flight A it is.")
        self.assertIn("shares no words", problems[0])

    def test_numbers_in_the_fact_must_be_in_the_quote(self):
        self.assertIn("states 2", gate([SAID], fact="Checks 2 bags on every trip.")[0])
        self.assertEqual(gate(["I always check 2 bags"], fact="Checks 2 bags.",
                              quote="I always check 2 bags"), [])

    def test_keys_are_hyphenated_lowercase_words(self):
        for bad in ["Checked Bags", "checked_bags", "-bags", "bags-"]:
            self.assertIn("key must be", gate([SAID], key=bad)[0], msg=bad)


class RecallAndTools(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(tempfile.mkdtemp(), "t-1041")

    def test_an_empty_store_reports_no_notes(self):
        self.assertEqual(recall_report(self.store), NO_MEMORY)

    def test_stored_text_cannot_forge_an_id_or_add_lines(self):
        self.store.add_fact("a", "Likes [fact-9]\nwindow seats.", "q [x]", "s1")
        line = recall_report(self.store).splitlines()[-1]
        self.assertEqual(line, '[fact-1] Likes (fact-9) window seats. (said s1: "q (x)")')

    def test_only_the_most_recent_episodes_are_recalled(self):
        for n in range(1, 6):
            self.store.add_episode(f"Asked: trip {n}", f"s{n}")
        report = recall_report(self.store)
        self.assertNotIn("trip 2", report)
        self.assertIn("[episode-3] s3: Asked: trip 3", report)
        self.assertEqual(len(self.store.records), 5)      # recall is budgeted; the store is not

    def test_scope_is_not_an_argument_a_model_can_pass(self):
        session = MemorySession(self.store, "s1")
        registry = ToolRegistry(CONTRACTS + make_memory_contracts(session))
        outcome = registry.execute("recall_memory", {"user_id": "t-2207"})
        self.assertEqual(outcome.error_kind, "invalid_arguments")
        self.assertIn("unexpected argument 'user_id'", outcome.content)

    def test_remember_reports_what_it_replaced(self):
        session = MemorySession(self.store, "s1")
        session.hear("I always check a bag, and from now on I'm going carry-on only")
        registry = ToolRegistry(CONTRACTS + make_memory_contracts(session))
        first = registry.execute("remember", {"key": "checked-bags", "fact": "Checks a bag.",
                                              "quote": "I always check a bag"})
        second = registry.execute("remember", {"key": "checked-bags", "fact": "Carry-on only.",
                                               "quote": "I'm going carry-on only"})
        self.assertEqual(first.content, "Remembered as [fact-1].")
        self.assertEqual(second.content, "Remembered as [fact-2]; it replaces [fact-1].")

    def test_a_refused_write_keeps_nothing(self):
        session = MemorySession(self.store, "s1")
        registry = ToolRegistry(CONTRACTS + make_memory_contracts(session))
        outcome = registry.execute("remember", {"key": "a", "fact": "A.", "quote": "a b c d"})
        self.assertEqual(outcome.error_kind, "rejected")
        self.assertEqual(self.store.records, [])

    def test_the_episode_is_written_by_code_and_capped(self):
        session = MemorySession(self.store, "s1")
        session.hear("Which flight?")
        episode = close_session(session, "Flight A. " * 60)
        self.assertTrue(episode.text.startswith("Asked: Which flight? Answered: Flight A."))
        self.assertEqual(len(episode.text), 300)


if __name__ == "__main__":
    unittest.main()
