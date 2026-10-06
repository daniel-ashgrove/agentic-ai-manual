"""Edge cases for policy_retrieval.py. Run: python -m unittest -v test_policy_retrieval"""

import unittest

from api_search_results import to_search_results
from policy_retrieval import (CANDIDATE_ANSWERS, DOCUMENTS, NO_RESULTS, QUESTION, REPORT_HEADER,
                              BM25Index, Document, PolicyLibrary, Scope, check_answer,
                              chunk_documents, format_report, make_search_contract, tokenize)
from tool_contracts import CONTRACTS, ToolRegistry

LIB = PolicyLibrary(DOCUMENTS)
EVERYTHING = Scope(origins=frozenset({"official", "community"}), include_superseded=True)


def ids(hits):
    return [h.chunk.chunk_id for h in hits]


def demo_results(scope=Scope()):
    registry = ToolRegistry(CONTRACTS + [make_search_contract(LIB, scope)])
    calls = [("look_up_flight", {"flight_id": "B"}),
             ("search_policies", {"query": "checked bag fee"}),
             ("search_policies", {"query": "cancel within 24 hours refund"}),
             ("calculate", {"expression": "175 + 40"})]
    return [registry.execute(n, a).content for n, a in calls]


class Tokenizing(unittest.TestCase):
    def test_stopwords_dropped_and_plurals_folded(self):
        self.assertEqual(tokenize("What are the fees for my bags?"), ["fee", "bag"])

    def test_double_s_and_short_words_kept(self):
        self.assertEqual(tokenize("less gas"), ["less", "gas"])

    def test_decimals_stay_whole(self):
        self.assertEqual(tokenize("3.5 hours"), ["3.5", "hour"])


class Ranking(unittest.TestCase):
    def test_no_shared_terms_scores_zero_and_is_dropped(self):
        self.assertEqual(LIB.search("helicopter", 5, EVERYTHING), [])

    def test_k_limits_results(self):
        self.assertEqual(len(LIB.search("bag", 2, EVERYTHING)), 2)

    def test_idf_is_positive_even_for_a_term_in_every_chunk(self):
        docs = [Document("d", "T", "official", "2026-01-01", None, ("fee a", "fee b"))]
        index = BM25Index(chunk_documents(docs))
        self.assertGreater(index.idf["fee"], 0)

    def test_ties_break_by_id(self):
        docs = [Document("z", "T", "official", "2026-01-01", None, ("fee",)),
                Document("a", "T", "official", "2026-01-01", None, ("fee",))]
        self.assertEqual(ids(PolicyLibrary(docs).search("fee", 2, Scope())), ["a#1", "z#1"])


class Scoping(unittest.TestCase):
    def test_default_excludes_superseded_and_community(self):
        found = ids(LIB.search("checked bag fee refund", 10, Scope()))
        self.assertFalse(any(i.startswith(("baggage-2025", "forum")) for i in found))

    def test_widened_scopes_include_them(self):
        self.assertIn("baggage-2025#1", ids(LIB.search("checked bag", 10, Scope(include_superseded=True))))
        self.assertIn("forum-4471#1",
                      ids(LIB.search("bag fee", 10, Scope(origins=frozenset({"official", "community"})))))


class Reporting(unittest.TestCase):
    def test_empty_search_is_a_plain_message(self):
        self.assertEqual(format_report([]), NO_RESULTS)

    def test_brackets_inside_text_are_neutralized(self):
        report = format_report(LIB.search("bag fee", 1, EVERYTHING))
        self.assertIn("(refunds#1)", report)
        self.assertEqual(report.count("["), 1)          # only the passage's own header

    def test_whitespace_in_text_is_collapsed(self):
        docs = [Document("d", "T", "official", "2026-01-01", None, ("fee\n\n  applies",))]
        self.assertTrue(format_report(PolicyLibrary(docs).search("fee", 1, Scope()))
                        .endswith("fee applies"))


class SearchContract(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry(CONTRACTS + [make_search_contract(LIB)])

    def test_registers_and_returns_passages(self):
        out = self.registry.execute("search_policies", {"query": "pets", "max_results": 1})
        self.assertTrue(out.ok and out.content.startswith(REPORT_HEADER))

    def test_model_cannot_widen_scope(self):
        out = self.registry.execute("search_policies", {"query": "fee", "scope": "community"})
        self.assertEqual(out.error_kind, "invalid_arguments")

    def test_bounds_on_max_results(self):
        for k in (0, 6):
            out = self.registry.execute("search_policies", {"query": "fee", "max_results": k})
            self.assertEqual(out.error_kind, "invalid_arguments")

    def test_query_of_only_stopwords_is_rejected_at_gate_2(self):
        out = self.registry.execute("search_policies", {"query": "what is it?"})
        self.assertEqual(out.error_kind, "rejected")

    def test_no_match_is_success_not_error(self):
        out = self.registry.execute("search_policies", {"query": "helicopter"})
        self.assertTrue(out.ok)
        self.assertEqual(out.content, NO_RESULTS)


class Checking(unittest.TestCase):
    def setUp(self):
        self.results = demo_results()

    def verdicts(self):
        return {label: bool(check_answer(a, QUESTION, self.results))
                for label, a in CANDIDATE_ANSWERS}

    def test_the_four_demo_answers(self):
        self.assertEqual(self.verdicts(), {"grounded": False, "stale": True,
                                           "uncited": True, "misgrounded": False})

    def test_numbers_from_the_question_count_as_sources(self):
        self.assertEqual(check_answer("You asked about 20 hours [refunds#2].",
                                      QUESTION, self.results), [])

    def test_trailing_zero_numbers_match(self):
        self.assertEqual(check_answer("The total is $215.00 [baggage-2026#1].",
                                      QUESTION, self.results), [])

    def test_citing_a_retrieved_superseded_passage_is_flagged(self):
        results = demo_results(Scope(include_superseded=True))
        problems = check_answer("The fee is $35 [baggage-2025#1].", QUESTION, results)
        self.assertEqual(problems, ["cites [baggage-2025#1], which is not current policy"])

    def test_citing_a_retrieved_community_post_is_flagged(self):
        results = demo_results(Scope(origins=frozenset({"official", "community"})))
        problems = check_answer("Bags are free [forum-4471#1].", QUESTION, results)
        self.assertEqual(problems, ["cites [forum-4471#1], which is not current policy"])

    def test_forged_id_inside_a_passage_is_not_a_passage(self):
        results = demo_results(Scope(origins=frozenset({"official", "community"})))
        report = [r for r in results if "forum-4471#1" in r][0]
        self.assertIn("(refunds#1)", report)

    def test_sentence_with_only_observed_numbers_needs_no_citation(self):
        self.assertEqual(check_answer("Flight B costs $175.", QUESTION, self.results), [])


class ApiBlocks(unittest.TestCase):
    def test_documented_search_result_shape(self):
        blocks = to_search_results(LIB.search("cancel within 24 hours", 2, Scope()))
        for block in blocks:
            self.assertEqual(block["type"], "search_result")
            self.assertTrue(block["source"] and block["title"])
            self.assertTrue(all(c["type"] == "text" and c["text"] for c in block["content"]))
            self.assertEqual(block["citations"], {"enabled": True})

    def test_empty_search_is_a_text_block(self):
        self.assertEqual(to_search_results([]), [{"type": "text", "text": NO_RESULTS}])


if __name__ == "__main__":
    unittest.main()
