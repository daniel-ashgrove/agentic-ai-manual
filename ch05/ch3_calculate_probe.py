"""Probe: Chapter 3's calculate() tool, copied verbatim, against inputs a model could plausibly send."""

def calculate(expression: str) -> float:
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        raise ValueError(f"Unsafe expression: {expression}")
    return eval(expression)

for expr in [
    "340 * 7 * 0.85",            # the happy path
    "340 * 7 * (1 - 15%)",       # natural way to write a discount
    "340 * 7 *",                 # truncated / malformed
    "340 * 7 / (3 - 3)",         # division by zero
    "9**9**9" ,                  # exponent allowed by the character filter
]:
    if expr == "9**9**9":
        print(f"{expr!r:28} -> (not executed here: the character filter accepts it)")
        # Only check the filter, don't run it.
        print("   passes character filter:", set(expr) <= set("0123456789+-*/(). "))
        continue
    try:
        print(f"{expr!r:28} -> {calculate(expr)!r}")
    except Exception as e:
        print(f"{expr!r:28} -> raises {type(e).__name__}: {e}")
