"""Check that your setup can run this book's code, before Chapter 1.

    python check_setup.py          checks Python, the packages and your settings: free
    python check_setup.py --live   also makes one very small call to the model

Every check prints OK, NOTE or FIX. FIX means something will not run until you change
it. NOTE means something is missing that only the programs calling a model need: everything
else in the book runs without it.
"""
import os
import sys
from importlib.metadata import PackageNotFoundError, version

# The versions the book's code was tested with. Newer versions usually work; if a
# listing behaves differently from the book, installing exactly these is the first fix.
TESTED = {"anthropic": "1.11.0", "langchain": "1.4.3", "langchain-core": "1.6.6",
          "langchain-anthropic": "1.7.5", "langgraph": "1.2.12",
          "opentelemetry-sdk": "1.45.0"}
PIP_LINE = "pip install " + " ".join(f"{name}=={v}" for name, v in TESTED.items())

problems = 0


def report(status: str, message: str) -> None:
    global problems
    problems += status == "FIX"
    print(f"  {status:4}  {message}")


def check_python() -> None:
    found = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info >= (3, 10):
        report("OK", f"Python {found}")
    else:
        report("FIX", f"Python {found}: the packages below need Python 3.10 or newer "
                      f"(a chapter's own demonstration runs on 3.9)")


def check_packages() -> None:
    missing = []
    for name, tested in TESTED.items():
        try:
            found = version(name)
        except PackageNotFoundError:
            missing.append(name)
            continue
        if found == tested:
            report("OK", f"{name} {found}")
        else:
            report("NOTE", f"{name} {found} (the book was tested with {tested})")
    if missing:
        report("FIX", f"not installed: {', '.join(missing)}. Install everything with:\n"
                      f"        {PIP_LINE}")


def check_settings() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        report("OK", "ANTHROPIC_API_KEY is set")
    else:
        report("NOTE", "ANTHROPIC_API_KEY is not set: needed only for programs that call a model")
    model = os.environ.get("BOOK_MODEL")
    if model:
        report("OK", f"BOOK_MODEL is {model}")
    else:
        report("NOTE", "BOOK_MODEL is not set: needed only for programs that call a model")


def check_live() -> None:
    if not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("BOOK_MODEL")):
        report("FIX", "--live needs both ANTHROPIC_API_KEY and BOOK_MODEL set")
        return
    try:
        import anthropic
        reply = anthropic.Anthropic().messages.create(
            model=os.environ["BOOK_MODEL"], max_tokens=256,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}])
    except Exception as exc:          # the API's own message says what is wrong
        report("FIX", f"the model call failed: {type(exc).__name__}: {exc}")
        return
    text = "".join(getattr(block, "text", "") for block in reply.content).strip()
    report("OK", f"{reply.model} answered {text[:30]!r}")


if __name__ == "__main__":
    print("Checking your setup:")
    check_python()
    check_packages()
    check_settings()
    if "--live" in sys.argv[1:]:
        check_live()
    print("\nReady." if not problems
          else f"\n{problems} thing(s) to fix before running the code.")
    sys.exit(1 if problems else 0)
