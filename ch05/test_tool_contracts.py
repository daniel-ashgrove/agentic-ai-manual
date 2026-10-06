"""Edge-case tests for tool_contracts.py. Run: python -m unittest -v test_tool_contracts"""

import logging
import unittest

import tool_contracts as tc
from tool_contracts import ContractError, ToolContract, ToolRegistry, ToolRejected

logging.getLogger("tool_contracts").addHandler(logging.NullHandler())
logging.getLogger("tool_contracts").propagate = False


def schema(**props):
    return {"type": "object", "properties": props,
            "required": list(props), "additionalProperties": False}


class ShapeGate(unittest.TestCase):
    def test_bool_is_not_an_integer(self):
        problems = tc.validate_args(schema(n={"type": "integer"}), {"n": True})
        self.assertEqual(problems, ["'n' must be an integer, got boolean True"])

    def test_numeric_string_is_not_coerced(self):
        problems = tc.validate_args(schema(n={"type": "integer"}), {"n": "2"})
        self.assertEqual(problems, ["'n' must be an integer, got string '2'"])

    def test_nan_and_infinity_are_rejected(self):
        s = schema(x={"type": "number", "minimum": 0, "maximum": 10})
        for bad in (float("nan"), float("inf")):
            self.assertEqual(tc.validate_args(s, {"x": bad}), ["'x' must be a finite number"])

    def test_bounds(self):
        s = schema(x={"type": "integer", "minimum": 1, "maximum": 3})
        self.assertEqual(tc.validate_args(s, {"x": 0}), ["'x' must be at least 1, got 0"])
        self.assertEqual(tc.validate_args(s, {"x": 4}), ["'x' must be at most 3, got 4"])
        self.assertEqual(tc.validate_args(s, {"x": 2}), [])

    def test_enum(self):
        s = schema(u={"type": "string", "enum": ["c", "f"]})
        self.assertEqual(tc.validate_args(s, {"u": "k"}), ["'u' must be one of ['c', 'f'], got 'k'"])

    def test_long_values_are_truncated_when_echoed(self):
        problems = tc.validate_args(schema(n={"type": "integer"}), {"n": "x" * 5000})
        self.assertLess(len(problems[0]), 100)


class RegistrationFailsLoudly(unittest.TestCase):
    def build(self, input_schema, name="t"):
        return ToolRegistry([ToolContract(name, "d", input_schema, handler=lambda **k: "ok")])

    def test_unsupported_property_keyword(self):
        for keyword, value in (("pattern", "^a$"), ("format", "date"), ("default", 1)):
            with self.assertRaises(ContractError):
                self.build(schema(a={"type": "string", keyword: value}))

    def test_unsupported_top_level_keyword(self):
        bad = schema(a={"type": "string"})
        bad["anyOf"] = []
        with self.assertRaises(ContractError):
            self.build(bad)

    def test_keyword_on_the_wrong_type_would_be_silently_ignored(self):
        with self.assertRaises(ContractError):
            self.build(schema(a={"type": "integer", "minLength": 2}))

    def test_open_schemas_are_refused(self):
        open_schema = schema(a={"type": "string"})
        del open_schema["additionalProperties"]
        with self.assertRaises(ContractError):
            self.build(open_schema)

    def test_required_must_name_a_declared_property(self):
        bad = schema(a={"type": "string"})
        bad["required"] = ["a", "b"]
        with self.assertRaises(ContractError):
            self.build(bad)

    def test_names(self):
        for bad_name in ("has space", "", "x" * 65, "dot.name"):
            with self.assertRaises(ContractError):
                self.build(schema(a={"type": "string"}), name=bad_name)
        good = ToolContract("t", "d", schema(a={"type": "string"}), handler=lambda a: a)
        with self.assertRaises(ContractError):
            ToolRegistry([good, good])


class ExecutionGate(unittest.TestCase):
    def registry(self, handler, check=None):
        return ToolRegistry([ToolContract("t", "d", schema(a={"type": "string"}),
                                          handler=handler, check=check)])

    def test_execute_never_raises(self):
        def boom(a):
            raise RuntimeError("secret-internal-detail")
        outcome = self.registry(boom).execute("t", {"a": "x"})
        self.assertEqual(outcome.error_kind, "tool_failed")
        self.assertNotIn("secret-internal-detail", outcome.content)

    def test_a_crashing_meaning_check_is_a_tool_failure(self):
        def bad_check(args):
            raise KeyError("oops-internal")
        outcome = self.registry(lambda a: a, check=bad_check).execute("t", {"a": "x"})
        self.assertEqual(outcome.error_kind, "tool_failed")
        self.assertNotIn("oops-internal", outcome.content)

    def test_expected_refusals_speak(self):
        def refuse(a):
            raise ToolRejected("sold out; try flight B")
        outcome = self.registry(refuse).execute("t", {"a": "x"})
        self.assertEqual(outcome.error_kind, "rejected")
        self.assertIn("sold out; try flight B", outcome.content)

    def test_handler_never_runs_when_a_gate_fails(self):
        calls = []
        registry = self.registry(lambda a: calls.append(a), check=lambda args: ["nope"])
        registry.execute("t", {"a": "x"})       # gate 2 fails
        registry.execute("t", {})               # gate 1 fails
        registry.execute("unknown", {"a": "x"})
        self.assertEqual(calls, [])

    def test_results_are_rendered_as_stable_json(self):
        outcome = self.registry(lambda a: {"b": 1, "a": 2}).execute("t", {"a": "x"})
        self.assertEqual(outcome.content, '{"a": 2, "b": 1}')

    def test_unserializable_result_is_a_tool_failure(self):
        outcome = self.registry(lambda a: object()).execute("t", {"a": "x"})
        self.assertEqual(outcome.error_kind, "tool_failed")

    def test_tool_result_block_shape(self):
        ok = tc.ToolOutcome(True, "42").as_tool_result("id1")
        bad = tc.ToolOutcome(False, "no", "rejected").as_tool_result("id2")
        self.assertEqual(ok, {"type": "tool_result", "tool_use_id": "id1", "content": "42"})
        self.assertIs(bad["is_error"], True)

    def test_unhashable_tool_name_is_handled(self):
        outcome = self.registry(lambda a: a).execute(["t"], {"a": "x"})
        self.assertEqual(outcome.error_kind, "unknown_tool")


class Calculator(unittest.TestCase):
    registry = ToolRegistry(tc.CONTRACTS)

    def run_expr(self, expression):
        return self.registry.execute("calculate", {"expression": expression})

    def test_surrounding_whitespace_is_tolerated(self):
        self.assertEqual(self.run_expr("  2 + 2  ").content, "4")

    def test_exponent_is_refused_before_any_evaluation(self):
        outcome = self.run_expr("9**9**9")
        self.assertEqual(outcome.error_kind, "rejected")
        self.assertIn("Pow", outcome.content)

    def test_non_finite_results_are_refused(self):
        for expr in ("1e308 * 10", "1e999", "1e308 + 1e308"):
            outcome = self.run_expr(expr)
            self.assertEqual(outcome.error_kind, "rejected", expr)
            self.assertIn("not a finite number", outcome.content)

    def test_large_integers_within_the_length_limit_are_fine(self):
        outcome = self.run_expr("9" * 98 + " * " + "9" * 98)
        self.assertTrue(outcome.ok)

    def test_names_strings_and_calls_are_refused(self):
        for expr in ("x + 1", "'a' * 3", "abs(-1)", "True + 1", "1 if 1 else 2", "[1]"):
            self.assertEqual(self.run_expr(expr).error_kind, "rejected", expr)

    def test_pathological_input_never_raises(self):
        for expr in ("(" * 99 + "1" + ")" * 99, "1\x00", "\u0661\u0662", "1 +", ")(", "1;2"):
            outcome = self.run_expr(expr)
            self.assertIsInstance(outcome, tc.ToolOutcome, repr(expr))

    def test_length_limit_applies_at_gate_1(self):
        outcome = self.run_expr("1+" * 101 + "1")
        self.assertEqual(outcome.error_kind, "invalid_arguments")

    def test_flight_id_is_normalized_but_not_guessed(self):
        r = self.registry
        self.assertTrue(r.execute("look_up_flight", {"flight_id": " c "}).ok)
        self.assertEqual(r.execute("look_up_flight", {"flight_id": "AB"}).error_kind, "rejected")


if __name__ == "__main__":
    unittest.main()
