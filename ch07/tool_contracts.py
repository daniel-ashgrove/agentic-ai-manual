"""
A tool-use contract layer, standard library only.

Every tool call a model makes passes through three gates before its result
goes back to the model:

    Gate 1  shape      does the call match the declared input schema?
    Gate 2  meaning    are the argument values acceptable in the real world?
    Gate 3  execution  did the tool run, and if not, what should the model hear?

Whatever happens, the model always receives a well-formed report.
"""

import ast
import json
import logging
import math
import operator
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

log = logging.getLogger("tool_contracts")


# --- Errors ---------------------------------------------------------------

class ContractError(Exception):
    """A contract is itself malformed. Raised at registration, never mid-run."""


class ToolRejected(Exception):
    """A tool raises this for an *expected* refusal. The message is written
    for the model to read, so it must say what went wrong and what to try."""


# --- Declaration: which schema features this layer understands -------------
# A validator that silently ignores a constraint it does not understand is
# worse than no validator. So anything outside this list fails at registration.

_TOP_KEYS = {"type", "properties", "required", "additionalProperties", "description"}
_PROP_KEYS = {"type", "enum", "minimum", "maximum", "minLength", "maxLength",
              "description"}
_SCALARS = {"string", "integer", "number", "boolean"}
_ONLY_FOR = {"minimum": {"integer", "number"}, "maximum": {"integer", "number"},
             "minLength": {"string"}, "maxLength": {"string"}}
_TOOL_NAME = re.compile(r"[a-zA-Z0-9_-]{1,64}")   # the Anthropic API's name rule


def check_schema(tool: str, schema: dict) -> None:
    where = f"contract {tool!r}"
    if set(schema) - _TOP_KEYS:
        raise ContractError(f"{where}: unsupported keyword(s) "
                            f"{sorted(set(schema) - _TOP_KEYS)}")
    if schema.get("type") != "object":
        raise ContractError(f"{where}: input_schema must have type 'object'")
    if schema.get("additionalProperties") is not False:
        raise ContractError(f"{where}: set additionalProperties to False")
    props = schema.get("properties", {})
    for name, spec in props.items():
        if set(spec) - _PROP_KEYS:
            raise ContractError(f"{where}: {name!r} uses unsupported keyword(s) "
                                f"{sorted(set(spec) - _PROP_KEYS)}")
        if spec.get("type") not in _SCALARS:
            raise ContractError(f"{where}: {name!r} needs a scalar type "
                                f"{sorted(_SCALARS)}")
        for keyword, allowed in _ONLY_FOR.items():
            if keyword in spec and spec["type"] not in allowed:
                raise ContractError(f"{where}: {keyword!r} does not apply to {name!r}")
    if set(schema.get("required", [])) - set(props):
        raise ContractError(f"{where}: 'required' names an undeclared property")


# --- Gate 1: shape ---------------------------------------------------------

_TYPE_WORDS = {"string": "a string", "integer": "an integer",
               "number": "a number", "boolean": "a boolean"}


def _type_name(value: Any) -> str:
    if isinstance(value, bool):      # bool is a subclass of int in Python,
        return "boolean"             # so it must be tested first
    for python_type, name in ((int, "integer"), (float, "number"), (str, "string"),
                              (dict, "object"), (list, "array")):
        if isinstance(value, python_type):
            return name
    return "null" if value is None else type(value).__name__


def _matches(value: Any, expected: str) -> bool:
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return _type_name(value) == expected


def _short(value: Any, limit: int = 40) -> str:
    """Echo untrusted values back to the model, but never at length."""
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def validate_args(schema: dict, args: Any) -> list[str]:
    """Return a list of problems written for the model. Empty means valid."""
    if not isinstance(args, dict):
        return [f"arguments must be a JSON object, got {_type_name(args)}"]
    props, problems = schema["properties"], []
    for name in schema.get("required", []):
        if name not in args:
            problems.append(f"missing required argument {name!r}")
    for name in sorted(set(args) - set(props), key=str):
        problems.append(f"unexpected argument {name!r} (allowed: {', '.join(props)})")
    for name, spec in props.items():
        if name not in args:
            continue
        value = args[name]
        if not _matches(value, spec["type"]):
            problems.append(f"{name!r} must be {_TYPE_WORDS[spec['type']]}, "
                            f"got {_type_name(value)} {_short(value)}")
        elif isinstance(value, float) and not math.isfinite(value):
            problems.append(f"{name!r} must be a finite number")
        elif "enum" in spec and value not in spec["enum"]:
            problems.append(f"{name!r} must be one of {spec['enum']}, "
                            f"got {_short(value)}")
        elif "minimum" in spec and value < spec["minimum"]:
            problems.append(f"{name!r} must be at least {spec['minimum']}, got {value}")
        elif "maximum" in spec and value > spec["maximum"]:
            problems.append(f"{name!r} must be at most {spec['maximum']}, got {value}")
        elif "minLength" in spec and len(value) < spec["minLength"]:
            problems.append(f"{name!r} must be at least {spec['minLength']} "
                            f"characters long")
        elif "maxLength" in spec and len(value) > spec["maxLength"]:
            problems.append(f"{name!r} must be at most {spec['maxLength']} "
                            f"characters long")
    return problems


# --- The contract, the report, and the registry ----------------------------

@dataclass(frozen=True)
class ToolContract:
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., Any]                          # Gate 3 runs this
    check: Optional[Callable[[dict], list[str]]] = None  # Gate 2 runs this


@dataclass(frozen=True)
class ToolOutcome:
    ok: bool
    content: str                       # exactly what the model will read
    # error_kind: unknown_tool | invalid_arguments | rejected | tool_failed
    error_kind: Optional[str] = None

    def as_tool_result(self, tool_use_id: str) -> dict:
        """The tool_result block the Anthropic Messages API expects."""
        block = {"type": "tool_result", "tool_use_id": tool_use_id,
                 "content": self.content}
        if not self.ok:
            block["is_error"] = True
        return block


def _render(result: Any) -> str:
    return result if isinstance(result, str) else json.dumps(result, sort_keys=True)


class ToolRegistry:
    def __init__(self, contracts: list[ToolContract]):
        self._by_name: dict[str, ToolContract] = {}
        for contract in contracts:
            if not _TOOL_NAME.fullmatch(contract.name):
                raise ContractError(f"invalid tool name {contract.name!r}")
            if contract.name in self._by_name:
                raise ContractError(f"duplicate tool name {contract.name!r}")
            check_schema(contract.name, contract.input_schema)
            self._by_name[contract.name] = contract

    def api_tools(self) -> list[dict]:
        """The declarations sent to the model in the API request's `tools`."""
        return [{"name": c.name, "description": c.description,
                 "input_schema": c.input_schema} for c in self._by_name.values()]

    def admit(self, name: Any, args: Any) -> Optional[ToolOutcome]:
        """Gates 1 and 2. Returns a failure report, or None if the call may run."""
        contract = self._by_name.get(name) if isinstance(name, str) else None
        if contract is None:
            return ToolOutcome(False, f"Unknown tool {_short(name, 64)}. Available "
                               f"tools: {', '.join(self._by_name)}.", "unknown_tool")
        problems, kind = validate_args(contract.input_schema, args), "invalid_arguments"
        if not problems and contract.check is not None:                # Gate 2
            kind = "rejected"
            try:
                problems = contract.check(args)
            except Exception:
                log.exception("meaning check for %r crashed", name)
                return self._failed(name)
        if problems:
            return ToolOutcome(False, f"Invalid arguments for {name!r}: "
                               f"{'; '.join(problems)}. Correct the arguments and "
                               f"call the tool again.", kind)
        return None

    def execute(self, name: Any, args: Any) -> ToolOutcome:
        """All three gates. Never raises; always returns a report."""
        refusal = self.admit(name, args)
        if refusal is not None:
            return refusal
        try:                                                           # Gate 3
            return ToolOutcome(True, _render(self._by_name[name].handler(**args)))
        except ToolRejected as expected:
            return ToolOutcome(False, f"{name!r} could not complete the request: "
                               f"{expected}", "rejected")
        except Exception:
            log.exception("tool %r failed unexpectedly", name)
            return self._failed(name)

    @staticmethod
    def _failed(name: str) -> ToolOutcome:
        # The details go to the log, not to the model.
        return ToolOutcome(False, f"Tool {name!r} failed with an internal error "
                           f"that is unrelated to your arguments. If a retry fails "
                           f"too, tell the user this tool is unavailable.",
                           "tool_failed")


# --- Two example contracts: the tools from Chapters 3 and 4, rebuilt -------

FLIGHTS = {
    "A": {"price_usd": 210, "duration_hours": 2.5, "arrival": "17:30"},
    "B": {"price_usd": 175, "duration_hours": 3.5, "arrival": "16:45"},
    "C": {"price_usd": 240, "duration_hours": 4.0, "arrival": "19:15"},
}


def _flight_key(flight_id: str) -> str:
    return flight_id.strip().upper()


def check_flight(args: dict) -> list[str]:
    """Gate 2: the set of real flights is data, not schema, so it lives here."""
    if _flight_key(args["flight_id"]) in FLIGHTS:
        return []
    return [f"no flight with id {args['flight_id']!r}; available flights: "
            f"{', '.join(FLIGHTS)}"]


def look_up_flight(flight_id: str) -> dict:
    return FLIGHTS[_flight_key(flight_id)]


_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _check_node(node: ast.AST) -> None:
    """Allow only numbers, + - * /, unary signs, and parentheses."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only plain numbers are allowed")
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        _check_node(node.operand)
    elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        _check_node(node.left)
        _check_node(node.right)
    elif isinstance(node, (ast.BinOp, ast.UnaryOp)):
        raise ValueError(f"the operator {type(node.op).__name__} is not allowed; "
                         f"use only numbers, + - * / and parentheses")
    else:
        raise ValueError(f"{type(node).__name__} is not allowed; use only numbers, "
                         f"+ - * / and parentheses")


def check_expression(args: dict) -> list[str]:
    """Gate 2: is this a well-formed expression in the allowed grammar?"""
    try:
        tree = ast.parse(args["expression"].strip(), mode="eval")
    except (SyntaxError, ValueError):
        return ["not a well-formed arithmetic expression. Write something like "
                "'340 * 7 * (1 - 0.15)', with percentages as decimals"]
    try:
        _check_node(tree.body)
    except ValueError as problem:
        return [str(problem)]
    return []


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp):
        return _UNARY[type(node.op)](_eval(node.operand))
    return _BINARY[type(node.op)](_eval(node.left), _eval(node.right))


def calculate(expression: str) -> float:
    tree = ast.parse(expression.strip(), mode="eval")
    try:
        result = _eval(tree.body)
    except ZeroDivisionError:
        raise ToolRejected("division by zero") from None
    if isinstance(result, float) and not math.isfinite(result):
        raise ToolRejected("the result is not a finite number")
    return result


CONTRACTS = [
    ToolContract(
        name="look_up_flight",
        description=("Look up one flight by its letter ID, for example 'B'. Returns "
                     "JSON with price_usd, duration_hours, and arrival (24-hour "
                     "clock). Call it once per flight; there is no bulk lookup."),
        input_schema={
            "type": "object",
            "properties": {"flight_id": {
                "type": "string", "minLength": 1, "maxLength": 8,
                "description": "The flight's letter ID, e.g. 'A'."}},
            "required": ["flight_id"],
            "additionalProperties": False,
        },
        check=check_flight,
        handler=look_up_flight,
    ),
    ToolContract(
        name="calculate",
        description=("Evaluate an arithmetic expression using numbers, + - * / and "
                     "parentheses, and return the numeric result. Percentages must be "
                     "written as decimals: 15% is 0.15. Exponents are not supported."),
        input_schema={
            "type": "object",
            "properties": {"expression": {
                "type": "string", "minLength": 1, "maxLength": 200,
                "description": "For example '340 * 7 * (1 - 0.15)'."}},
            "required": ["expression"],
            "additionalProperties": False,
        },
        check=check_expression,
        handler=calculate,
    ),
]


# --- Demonstration: a scripted "model" that makes good and bad calls -------

def flaky_lookup(flight_id: str) -> dict:
    """A stand-in for an upstream outage. Its message names an internal host,
    which is exactly the kind of detail the model must never be shown."""
    raise ConnectionError("timeout talking to db-7.internal:5432")


def run_demo() -> None:
    captured: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    log.addHandler(Capture())
    log.propagate = False

    registry = ToolRegistry(CONTRACTS + [ToolContract(
        name="flaky_lookup",
        description="A test double that always fails.",
        input_schema={"type": "object",
                      "properties": {"flight_id": {"type": "string"}},
                      "required": ["flight_id"], "additionalProperties": False},
        handler=flaky_lookup)])

    scripted_calls = [
        ("valid call",             "calculate",      {"expression": "340 * 7 * (1 - 0.15)"}),
        ("valid, id normalized",   "look_up_flight", {"flight_id": "b"}),
        ("missing argument",       "look_up_flight", {}),
        ("wrong type",             "calculate",      {"expression": 340}),
        ("unexpected argument",    "look_up_flight", {"flight_id": "A", "seat": "12C"}),
        ("no such flight",         "look_up_flight", {"flight_id": "D"}),
        ("percent sign",           "calculate",      {"expression": "340 * 7 * (1 - 15%)"}),
        ("disallowed structure",   "calculate",      {"expression": "__import__('os').getcwd()"}),
        ("division by zero",       "calculate",      {"expression": "340 / (3 - 3)"}),
        ("unknown tool",           "book_flight",    {"flight_id": "B"}),
        ("arguments not an object", "calculate",     "340 * 7"),
        ("upstream crash",         "flaky_lookup",   {"flight_id": "A"}),
    ]

    print("=== Scripted calls ===")
    for label, name, args in scripted_calls:
        outcome = registry.execute(name, args)
        status = "ok" if outcome.ok else f"ERROR ({outcome.error_kind})"
        print(f"\n[{label}] {name}({args!r})")
        print(f"  {status}: {outcome.content}")

    print("\n=== The tool_result block sent back for the 'no such flight' call ===")
    failed = registry.execute("look_up_flight", {"flight_id": "D"})
    print(json.dumps(failed.as_tool_result("toolu_example"), indent=2))

    print("\n=== What was logged (never shown to the model) ===")
    for record in captured:
        print(f"{record.getMessage()} -> {record.exc_info[0].__name__}: {record.exc_info[1]}")


if __name__ == "__main__":
    run_demo()
