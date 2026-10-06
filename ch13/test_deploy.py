"""Tests the three deployment decisions: what one run is when many are in flight,
what the service writes down, and what the numbers are allowed to say.

Run: python -m unittest -v test_deploy
"""

import json
import logging
import tempfile
import threading
import unittest

import dashboard as w
import deploy as d
import telemetry as t
from agent_memory import recall_report
from combined import Stores
from guardrails import Ledger, Limits
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from policy_retrieval import QUESTION

logging.getLogger("contracts").setLevel(logging.CRITICAL)

DAY, LATER = d.DAY_ONE, d.DAY_TWO


def service_in(folder, **kwargs):
    clock = d.StepClock()
    log = d.EventLog(f"{folder}/events.jsonl", d.Release(), clock)
    service = d.Service(folder, log, Ledger(), d.scripted_engine(d.SCRIPT),
                        ids=d.counting_ids(), now=clock, **kwargs)
    return service, log, clock


class Serving(unittest.TestCase):
    """Serve: what belongs to one run, and what is shared on purpose."""

    def test_each_run_gets_its_own_identifier_and_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            service, log, _ = service_in(folder)
            replies = [service.answer(d.Request(QUESTION, "T-88", DAY))
                       for _ in range(3)]
            self.assertEqual(len({r.run for r in replies}), 3)
            answered = [e for e in log.read() if e["event"] == "answered"]
            self.assertEqual([e["calls"] for e in answered], [8, 8, 8])

    def test_concurrent_runs_do_not_share_a_budget_or_an_action(self):
        with tempfile.TemporaryDirectory() as folder:
            service, log, _ = service_in(folder)
            asks = [("T-88", DAY, QUESTION), ("T-63", LATER, d.CANCEL_Q)] * 4
            done = []

            def one(traveller, day, question):
                done.append(service.answer(d.Request(question, traveller, day)))

            threads = [threading.Thread(target=one, args=a) for a in asks]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(len({reply.run for reply in done}), 8)
            answered = [e for e in log.read() if e["event"] == "answered"]
            self.assertEqual(len(answered), 8)
            self.assertTrue(all(e["calls"] <= e["ceiling"] for e in answered))
            self.assertEqual(sorted(len(r.held) for r in done), [0] * 4 + [1] * 4)

    def test_one_travellers_memory_is_not_another_travellers(self):
        with tempfile.TemporaryDirectory() as folder:
            service, _, _ = service_in(folder)
            service.answer(d.Request(d.QUOTED_Q, "T-41", DAY))
            service.answer(d.Request(QUESTION, "T-88", DAY))
            quoted = recall_report(Stores(folder, "T-41").memory())
            other = recall_report(Stores(folder, "T-88").memory())
            self.assertIn("$190", quoted)
            self.assertNotIn("$190", other)

    def test_the_ledger_is_shared_so_a_rate_limit_survives_a_second_request(self):
        with tempfile.TemporaryDirectory() as folder:
            service, _, _ = service_in(folder)
            first = service.answer(d.Request(d.CANCEL_Q, "T-63", LATER))
            service.decide(first.run, "dana", approve=True)
            self.assertEqual(len(service.ledger.applied), 1)
            self.assertEqual(service.ledger.count("T-63", LATER), 1)

    def test_a_run_that_raises_leaves_the_service_up(self):
        with tempfile.TemporaryDirectory() as folder:
            service, log, _ = service_in(folder)
            broken = service.answer(d.Request("a question no stand-in knows", "T-1", DAY))
            self.assertFalse(broken.delivered)
            self.assertEqual(broken.halted, "service-error")
            self.assertEqual(broken.answer, d.SORRY)
            self.assertEqual([e["error"] for e in log.read() if e["event"] == "failed"],
                             ["KeyError"])
            self.assertTrue(service.answer(d.Request(QUESTION, "T-88", DAY)).delivered)

    def test_the_deadline_refuses_the_next_delegation_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            service, log, _ = service_in(folder, deadline_s=2.0)
            reply = service.answer(d.Request(d.CANCEL_Q, "T-52", LATER))
            refused = [e for e in log.read() if e["event"] == "deadline"]
            self.assertEqual([e["worker"] for e in refused], ["fares"])
            self.assertTrue(reply.delivered)          # delivered, and worse for it
            self.assertNotIn("$125", reply.answer)

    def test_a_generous_deadline_refuses_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            service, log, _ = service_in(folder, deadline_s=60.0)
            service.answer(d.Request(d.CANCEL_Q, "T-63", LATER))
            self.assertEqual([e for e in log.read() if e["event"] == "deadline"], [])

    def test_health_answers_without_running_anything(self):
        with tempfile.TemporaryDirectory() as folder:
            service, log, _ = service_in(folder)
            server = d.serve(service, port=0)
            port = server.server_address[1]
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                import urllib.request
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as up:
                    self.assertEqual(json.loads(up.read())["ok"], True)
            finally:
                server.shutdown()
                server.server_close()
            self.assertEqual(log.read(), [])


class Emitting(unittest.TestCase):
    """Emit: one line per decision, and the record kept as a person reads it."""

    def test_every_run_has_one_received_and_one_answered(self):
        with tempfile.TemporaryDirectory() as folder:
            service, log, _ = service_in(folder)
            reply = service.answer(d.Request(QUESTION, "T-88", DAY))
            events = [e for e in log.read() if e["run"] == reply.run]
            self.assertEqual([e["event"] for e in events], ["received", "answered"])

    def test_the_answered_event_carries_chapter_elevens_record_verbatim(self):
        with tempfile.TemporaryDirectory() as folder:
            service, log, _ = service_in(folder)
            service.answer(d.Request(d.CANCEL_Q, "T-63", LATER))
            answered = [e for e in log.read() if e["event"] == "answered"][0]
            self.assertTrue(any("held" in line and "$125 on BK-4471" in line
                                for line in answered["record"]))

    def test_every_event_names_the_release_it_came_from(self):
        with tempfile.TemporaryDirectory() as folder:
            service, log, _ = service_in(folder)
            service.answer(d.Request(QUESTION, "T-88", DAY))
            for event in log.read():
                self.assertEqual((event["service"], event["version"], event["model"]),
                                 ("flight-agent", "1.0.0", "scripted-stand-in"))

    def test_concurrent_writes_leave_whole_lines(self):
        with tempfile.TemporaryDirectory() as folder:
            service, log, _ = service_in(folder)
            threads = [threading.Thread(
                target=service.answer,
                args=(d.Request(QUESTION, "T-88", DAY),)) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            with open(f"{folder}/events.jsonl", encoding="utf-8") as written:
                lines = [line for line in written if line.strip()]
            self.assertEqual(len(lines), 16)
            for line in lines:                       # every one parses on its own
                json.loads(line)

    def test_a_persons_decision_lands_next_to_the_run_it_belongs_to(self):
        with tempfile.TemporaryDirectory() as folder:
            service, log, _ = service_in(folder)
            reply = service.answer(d.Request(d.CANCEL_Q, "T-63", LATER))
            service.decide(reply.run, "dana", approve=False, why="already refunded")
            service.resolve(reply.run, "dana", "told them by email")
            events = [e["event"] for e in log.read() if e["run"] == reply.run]
            self.assertEqual(events, ["received", "held", "answered",
                                      "reviewed", "resolved"])


class Watching(unittest.TestCase):
    """Watch: the five questions, and what makes each one an alert."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        service, log, clock = service_in(self.folder.name)
        held = service.answer(d.Request(d.CANCEL_Q, "T-63", LATER))
        service.answer(d.Request(d.QUOTED_Q, "T-41", DAY))
        halted = service.answer(d.Request(d.CREDIT_Q, "T-41", LATER))
        clock.skip(48)
        service.decide(held.run, "dana", approve=True)
        service.decide(halted.run, "raj", approve=False, why="no record of that quote")
        service.resolve(halted.run, "raj", "answered by hand")
        self.figures = w.summarise(log.read())

    def tearDown(self):
        self.folder.cleanup()

    def test_the_five_questions_are_answered_from_the_log_alone(self):
        self.assertEqual((self.figures["runs"], self.figures["delivered"],
                          self.figures["halted"]), (3, 2, 1))
        self.assertEqual(self.figures["halts"], {"untraceable-amount": 1})
        self.assertEqual(self.figures["approval_rate"], 0.5)
        self.assertEqual(self.figures["escalated"], 1)
        self.assertGreater(self.figures["worst_wait"], 47)
        self.assertEqual(self.figures["ceiling"], 16)

    def test_a_slow_reply_and_a_low_approval_rate_each_say_what_to_do(self):
        said = w.alerts(self.figures)
        self.assertTrue(any("waited 48 minutes" in line for line in said))
        self.assertTrue(any("Only 50%" in line for line in said))

    def test_approving_everything_is_an_alert_too(self):
        figures = dict(self.figures, approval_rate=1.0, worst_wait=1.0, out_of_time=0)
        self.assertEqual([line for line in w.alerts(figures) if "100%" in line],
                         [w.alerts(figures)[0]])
        self.assertIn("not a control", w.alerts(figures)[0])

    def test_an_unanswered_escalation_is_counted_even_with_nothing_to_time(self):
        figures = dict(self.figures, escalated=3, open=2, worst_wait=None,
                       median_wait=None)
        self.assertTrue(any("no reply at all" in line for line in w.alerts(figures)))

    def test_the_page_shows_the_numbers_it_was_given(self):
        rendered = w.page(self.figures, w.alerts(self.figures))
        self.assertIn("untraceable-amount", rendered)
        self.assertIn("50%", rendered)
        self.assertIn("2 delivered, 1 halted", rendered)


class Exporting(unittest.TestCase):
    """The same events, in a vocabulary another tool already understands."""

    def spans_for(self, events):
        exporter = InMemorySpanExporter()
        t.export(events, t.tracer_for(exporter))
        return exporter.get_finished_spans()

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        service, log, _ = service_in(self.folder.name)
        service.answer(d.Request(d.QUOTED_Q, "T-41", DAY))
        service.answer(d.Request(d.CREDIT_Q, "T-41", LATER))
        self.events = log.read()
        self.spans = self.spans_for(self.events)

    def tearDown(self):
        self.folder.cleanup()

    def test_a_run_is_an_invoke_agent_span_and_an_action_is_an_execute_tool_span(self):
        self.assertEqual(sorted({s.name for s in self.spans}),
                         ["execute_tool request_credit", "invoke_agent flight-agent"])
        for span in self.spans:
            self.assertIn(span.attributes["gen_ai.operation.name"],
                          ("invoke_agent", "execute_tool"))
            self.assertTrue(span.attributes["gen_ai.provider.name"])

    def test_a_halted_run_is_not_an_error(self):
        halted = [s for s in self.spans
                  if s.attributes.get("flight.run.halt_condition")]
        self.assertEqual(len(halted), 1)
        self.assertEqual(halted[0].attributes["flight.run.halt_condition"],
                         "untraceable-amount")
        self.assertEqual(halted[0].status.status_code.name, "UNSET")
        self.assertIsNone(halted[0].status.description)

    def test_the_conversation_id_is_the_session_the_application_manages(self):
        runs = [s for s in self.spans if s.name.startswith("invoke_agent")]
        self.assertEqual(sorted(s.attributes["gen_ai.conversation.id"] for s in runs),
                         [f"T-41:{DAY}", f"T-41:{LATER}"])

    def test_an_unfinished_run_is_not_a_span_yet(self):
        started = [e for e in self.events if e["event"] == "received"][:1]
        self.assertEqual(self.spans_for(started), ())

    def test_the_action_span_is_a_child_of_the_run_that_asked_for_it(self):
        run = [s for s in self.spans if s.name.startswith("invoke_agent")
               and s.attributes["flight.run.actions_held"] == 1][0]
        action = [s for s in self.spans if s.name.startswith("execute_tool")][0]
        self.assertEqual(action.parent.span_id, run.context.span_id)
        self.assertEqual(action.attributes["gen_ai.tool.name"], "request_credit")


if __name__ == "__main__":
    unittest.main(verbosity=2)
