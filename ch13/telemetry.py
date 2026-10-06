"""
The same events, as OpenTelemetry spans, so that a monitoring tool you did not
write can read them.

    pip install opentelemetry-sdk

dashboard.py answers five questions about one service. This answers none of
them, and that is the point: it puts the run on the same timeline as the
database call, the queue and the HTTP request in front of it, in a vocabulary
every tool in that class already understands.

The vocabulary is OpenTelemetry's semantic conventions for generative AI,
which give an agent run the operation name `invoke_agent`, a tool call
`execute_tool`, and attributes under `gen_ai.`. The conventions are marked
Development: names move between releases, so they are written here as the
literal strings this file was checked against rather than imported from a
constant that may be renamed.

Everything this service knows that the conventions do not cover — a halt
condition, an amount held, a reserve — goes under `flight.`, a namespace
nobody else owns. An invented `gen_ai.` attribute is a name collision waiting
for the release that defines it differently.
"""

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, use_span

OPERATION, PROVIDER = "gen_ai.operation.name", "gen_ai.provider.name"
AGENT, VERSION = "gen_ai.agent.name", "gen_ai.agent.version"
MODEL, CONVERSATION = "gen_ai.request.model", "gen_ai.conversation.id"
TOOL = "gen_ai.tool.name"
SECOND = 1_000_000_000                       # spans are timed in nanoseconds


def tracer_for(exporter, service: str = "flight-agent"):
    """A tracer that sends finished spans straight to `exporter`. A deployment
    swaps the exporter for an OTLP one and changes nothing else in this file."""
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("flight-agent.deploy")


def by_run(events: list) -> dict:
    runs = {}
    for event in events:
        runs.setdefault(event["run"], []).append(event)
    return runs


def export(events: list, tracer) -> int:
    """One span per run, with a child span for each action the run asked for.

    A halted run is not an error. It is the guardrail layer doing exactly what
    Chapter 11 built it to do, and a span status that says otherwise turns
    every safeguard into something an on-call engineer is paid to make stop.
    The halt condition is an attribute; the status stays unset.
    """
    exported = 0
    for run, timeline in by_run(events).items():
        start = next((e for e in timeline if e["event"] == "received"), None)
        end = next((e for e in timeline if e["event"] == "answered"), None)
        if not (start and end):
            continue                     # a run still in flight is not a span yet
        span = tracer.start_span(
            f"invoke_agent {start['service']}", kind=SpanKind.CLIENT,
            start_time=int(start["at"] * SECOND),
            attributes={
                OPERATION: "invoke_agent",
                PROVIDER: start.get("provider", "none.scripted"),
                AGENT: start["service"],
                VERSION: start["version"],
                MODEL: start["model"],
                # The application manages this conversation, so it owns the id.
                CONVERSATION: f"{start['traveller']}:{start['day']}",
                "flight.run.id": run,
                "flight.run.delivered": end["delivered"],
                "flight.run.halt_condition": end["halted"] or "",
                "flight.run.model_calls": end["calls"],
                "flight.run.model_call_ceiling": end["ceiling"],
                "flight.run.attempts": end["attempts"],
                "flight.run.actions_held": end["held"],
                "flight.run.out_of_time": any(e["event"] == "deadline"
                                              for e in timeline),
            })
        with use_span(span, end_on_exit=False):
            for action in (e for e in timeline if e["event"] == "held"):
                tracer.start_span(
                    "execute_tool request_credit", kind=SpanKind.INTERNAL,
                    start_time=int(action["at"] * SECOND),
                    attributes={OPERATION: "execute_tool",
                                PROVIDER: start.get("provider", "none.scripted"),
                                TOOL: "request_credit",
                                "flight.action.booking": action["booking"],
                                "flight.action.amount_usd": action["amount_usd"],
                                "flight.action.issued": False},
                ).end(end_time=int(action["at"] * SECOND))
        span.end(end_time=int(end["at"] * SECOND))
        exported += 1
    return exported


def show(spans) -> None:
    """What went out, in the order it was finished. Trace and span identifiers
    are random and are left out so this prints the same thing every time."""
    for span in spans:
        millis = (span.end_time - span.start_time) / 1_000_000
        print(f"  {span.name}  [{span.kind.name}]  {millis:.0f} ms")
        for key in sorted(span.attributes):
            print(f"      {key} = {span.attributes[key]!r}")


if __name__ == "__main__":
    from deploy import FOLDER, EventLog, Release

    events = EventLog(f"{FOLDER}/events.jsonl", Release()).read()
    exporter = InMemorySpanExporter()
    count = export(events, tracer_for(exporter))
    spans = exporter.get_finished_spans()
    print(f"=== {count} runs exported as {len(spans)} spans; two of them ===")
    show([s for s in spans if s.attributes.get("flight.run.id") in ("r-0003", "r-0005")])
