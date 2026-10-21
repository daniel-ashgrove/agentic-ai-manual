"""Tests for react_loop.py that need no API key: a scripted stand-in replaces the
Anthropic client, so the loop's own logic runs exactly as written. The anthropic
package must be installed, because react_loop.py imports it, but it is never called.

Run with:  python -m unittest -v test_react_loop
"""
import contextlib
import io
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("BOOK_MODEL", "scripted-stand-in")
import react_loop as r  # noqa: E402  (BOOK_MODEL must be set before the import)


def text(t):
    return SimpleNamespace(type="text", text=t)


def tool_use(expression, call_id="toolu_1"):
    return SimpleNamespace(type="tool_use", id=call_id, name="calculate",
                           input={"expression": expression})


def reply(stop_reason, *blocks):
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


class ScriptedClient:
    """Answers each messages.create call with the next scripted reply and keeps
    a copy of every request, so a test can check what the loop sent."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []
        self.messages = self

    def create(self, **request):
        self.requests.append({**request, "messages": list(request["messages"])})
        if not self.replies:
            raise AssertionError("the loop asked for more replies than were scripted")
        return self.replies.pop(0)


def run(client, question="What do I pay?", **kwargs):
    original, r.client = r.client, client
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return r.run_react_loop(question, **kwargs)
    finally:
        r.client = original


class Loop(unittest.TestCase):

    def test_a_tool_result_goes_back_with_the_call_id(self):
        first = reply("tool_use", text("I'll compute the discounted total."),
                      tool_use("340 * 7 * 0.85"))
        client = ScriptedClient(first, reply("end_turn", text("You pay $2,023.00.")))
        self.assertEqual(run(client), "You pay $2,023.00.")
        self.assertEqual(len(client.requests), 2)
        sent = client.requests[1]["messages"]
        # The assistant turn goes back exactly as it arrived, tool_use block included.
        self.assertEqual(sent[1], {"role": "assistant", "content": first.content})
        result = sent[2]["content"][0]
        self.assertEqual(result["type"], "tool_result")
        self.assertEqual(result["tool_use_id"], "toolu_1")
        self.assertEqual(float(result["content"]), 2023.0)

    def test_text_prints_with_a_tool_call_and_the_answer_is_returned_once(self):
        client = ScriptedClient(
            reply("tool_use", text("I'll compute the discounted total."),
                  tool_use("340 * 7 * 0.85")),
            reply("end_turn", text("You pay $2,023.00.")))
        original, r.client, out = r.client, client, io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                answer = r.run_react_loop("What do I pay?")
        finally:
            r.client = original
        self.assertEqual(answer, "You pay $2,023.00.")
        self.assertEqual(out.getvalue().splitlines(), [
            "Model response (step 1): "
            "I'll compute the discounted total.",
            "Action (step 1): calculate('340 * 7 * 0.85')",
            "Observation (step 1): 2023.0"])

    def test_the_request_names_one_tool_and_one_call_per_turn(self):
        client = ScriptedClient(reply("end_turn", text("Done.")))
        run(client)
        request = client.requests[0]
        self.assertEqual(request["model"], r.MODEL)
        self.assertEqual(request["tools"], [r.CALCULATOR_TOOL])
        self.assertTrue(request["tool_choice"]["disable_parallel_tool_use"])

    def test_any_other_stop_reason_is_an_early_stop(self):
        client = ScriptedClient(reply("max_tokens", text("The total is")))
        self.assertEqual(run(client), "Stopped early (stop_reason='max_tokens') "
                                      "before reaching a final answer.")

    def test_the_step_budget_ends_a_run_that_never_answers(self):
        client = ScriptedClient(reply("tool_use", tool_use("1 + 1")),
                                reply("tool_use", tool_use("2 + 2", "toolu_2")))
        self.assertEqual(run(client, max_steps=2),
                         "Could not reach a final answer within the step budget.")
        self.assertEqual(len(client.requests), 2)

    def test_a_tool_failure_ends_the_run(self):
        # Nothing catches it: the exception leaves the loop, and the model is never told.
        client = ScriptedClient(reply("tool_use", tool_use("340 * 7 * (1 - 15%)")))
        with self.assertRaises(ValueError):
            run(client)
        self.assertEqual(len(client.requests), 1)


class Calculator(unittest.TestCase):

    def test_the_filter_limits_characters_not_expressions(self):
        self.assertEqual(r.calculate("2**8"), 256)

    def test_a_character_outside_the_set_is_refused(self):
        with self.assertRaises(ValueError):
            r.calculate("15%")


if __name__ == "__main__":
    unittest.main()
