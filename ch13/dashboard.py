"""
Watch: the questions the event log has to answer, and a page that answers them.

A dashboard is not a decoration on top of a log. It is the list of questions
you decided, before anything went wrong, that you would need answered — and
every number below exists because some earlier chapter left a decision that
nobody could check from the outside:

    delivered / halted / by name   Chapter 11 halts instead of failing silently.
                                   How often, and on which condition?
    approval rate                  Chapter 11 puts a person in front of every
                                   action. Are they deciding, or clicking?
    escalation to reply            A halt is only a safeguard if somebody
                                   answers it. How long does that take?
    model calls per run            Chapter 11's ceiling and Chapter 12's
                                   reserve are guesses until traffic prices them.
    runs out of time               The one number no earlier chapter could have:
                                   a deadline exists only in a deployment.

Nothing here writes. It reads the log deploy.py wrote and answers those five
questions, which is why it can run in a different process, on a copy, or a
week later.
"""

import html
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class Watch:
    """What counts as bad. These are decisions, not defaults: each one says
    what somebody promised to do about the runs on the wrong side of it."""
    reply_minutes: float = 30.0        # how long an escalated run may wait
    approval_floor: float = 0.6        # below this, approval is not review
    approval_ceiling: float = 0.95     # above this, approval is not review either
    calls_share: float = 0.8           # of the ceiling, averaged over runs


def summarise(events: list) -> dict:
    """Five questions, one pass over the log."""
    answered = [e for e in events if e["event"] == "answered"]
    escalated = {e["run"]: e["at"] for e in events if e["event"] == "escalated"}
    resolved = {e["run"]: e["at"] for e in events if e["event"] == "resolved"}
    reviewed = [e for e in events if e["event"] == "reviewed"]
    held = [e for e in events if e["event"] == "held"]
    waits = sorted((resolved[run] - at) / 60 for run, at in escalated.items()
                   if run in resolved)
    approved = [e for e in reviewed if e["approved"]]
    ceiling = max([e["ceiling"] for e in answered], default=0)
    calls = [e["calls"] for e in answered]

    halts = {}
    for event in answered:
        if event["halted"]:
            halts[event["halted"]] = halts.get(event["halted"], 0) + 1
    return {
        "runs": len(answered),
        "delivered": sum(e["delivered"] for e in answered),
        "halted": sum(not e["delivered"] for e in answered),
        "failed": sum(e["event"] == "failed" for e in events),
        "halts": dict(sorted(halts.items())),
        "requested": len(held),
        "decided": len(reviewed),
        "approved": len(approved),
        "refused": len(reviewed) - len(approved),
        "waiting": len(held) - len(reviewed),
        "approval_rate": len(approved) / len(reviewed) if reviewed else None,
        "escalated": len(escalated),
        "open": len(escalated) - len(waits),
        "median_wait": statistics.median(waits) if waits else None,
        "worst_wait": max(waits) if waits else None,
        "calls_mean": statistics.mean(calls) if calls else 0,
        "calls_max": max(calls, default=0),
        "ceiling": ceiling,
        "out_of_time": len({e["run"] for e in events if e["event"] == "deadline"}),
        "releases": sorted({f"{e['service']} {e['version']} / {e['model']}"
                            for e in events}),
    }


def alerts(figures: dict, watch: Watch = Watch()) -> list:
    """What a person does when a number is bad. A threshold with no sentence
    after it is a decoration."""
    said = []
    rate, waited = figures["approval_rate"], figures["worst_wait"]
    if figures["failed"]:
        said.append(f"{figures['failed']} run(s) failed inside the service: read the "
                    f"error events before anything else on this page.")
    if waited is not None and waited > watch.reply_minutes:
        said.append(f"An escalated run waited {waited:.0f} minutes for a person "
                    f"({watch.reply_minutes:.0f} is the promise). Either staff the "
                    f"queue or stop promising a same-day answer.")
    if figures["open"]:
        said.append(f"{figures['open']} escalated run(s) have no reply at all. These "
                    f"are travellers who were told a colleague would come back.")
    if rate is not None and rate > watch.approval_ceiling:
        said.append(f"{rate:.0%} of requests were approved. A reviewer who approves "
                    f"everything is not a control; find out whether they are reading.")
    if rate is not None and rate < watch.approval_floor:
        said.append(f"Only {rate:.0%} of requests were approved. The agent is asking "
                    f"for things it should not ask for; the refusals say which.")
    if (figures["ceiling"]
            and figures["calls_mean"] > watch.calls_share * figures["ceiling"]):
        said.append(f"Runs average {figures['calls_mean']:.1f} of {figures['ceiling']} "
                    f"model calls. The next hard question halts on the budget.")
    if figures["out_of_time"]:
        said.append(f"{figures['out_of_time']} run(s) ran out of time and answered "
                    f"with less than they meant to. These are delivered runs; nothing "
                    f"in the outcome says they were worse.")
    return said


# --- The page ----------------------------------------------------------------

CARDS = [("Runs", "{runs}", "{delivered} delivered, {halted} halted"),
         ("Approved", "{approval}", "{approved} of {decided} decided, {waiting} waiting"),
         ("Slowest reply", "{worst}", "median {median}, {open} still open"),
         ("Model calls", "{calls_mean:.1f}", "of {ceiling}, worst run {calls_max}"),
         ("Out of time", "{out_of_time}", "answered with less than they meant to")]

STYLE = """
:root { --ink:#1A1A1A; --blue:#1E3A8A; --red:#DC2626; --amber:#FEF3C7;
        --line:#D8DCE6; --paper:#FFFFFF; }
* { box-sizing: border-box; }
body { font: 22px/1.5 Georgia, 'Times New Roman', serif; color: var(--ink);
       background: var(--paper); margin: 0; padding: 34px 38px; width: 1000px; }
h1 { font-size: 30px; color: var(--blue); margin: 0 0 4px; letter-spacing: .2px; }
.sub { font-size: 20px; color: #55607A; margin: 0 0 26px; }
.cards { display: flex; gap: 14px; margin-bottom: 26px; }
.card { flex: 1; border: 1px solid var(--line); border-top: 4px solid var(--blue);
        padding: 14px 16px; }
.card .label { font-size: 20px; color: #55607A; }
.card .value { font-size: 40px; color: var(--blue); line-height: 1.1; margin: 2px 0; }
.card .note { font-size: 20px; color: #55607A; }
h2 { font-size: 23px; color: var(--blue); margin: 0 0 10px; }
.row { display: flex; gap: 22px; }
.row > section { flex: 1; }
table { border-collapse: collapse; width: 100%; font-size: 20px; }
th { background: var(--blue); color: #FFF; text-align: left; font-weight: bold;
     padding: 7px 10px; }
td { border-bottom: 1px solid var(--line); padding: 7px 10px; }
td.n { text-align: right; font-family: Consolas, 'Courier New', monospace; }
ul.alerts { list-style: none; margin: 0 0 26px; padding: 0; }
ul.alerts li { border-left: 5px solid var(--red); background: #FDF2F2;
               padding: 9px 14px; margin-bottom: 8px; font-size: 20px; }
.calm { border-left: 5px solid var(--blue); background: var(--amber);
        padding: 9px 14px; font-size: 20px; }
footer { margin-top: 26px; font-size: 20px; color: #55607A;
         border-top: 1px solid var(--line); padding-top: 12px; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 19px; }
"""


def _minutes(value) -> str:
    return "—" if value is None else f"{value:.0f} min"


def page(figures: dict, said: list, watch: Watch = Watch()) -> str:
    """One screen. Everything on it is a number some earlier chapter's decision
    made checkable, and nothing on it is here because it was easy to plot."""
    filled = dict(figures,
                  approval=("—" if figures["approval_rate"] is None
                            else f"{figures['approval_rate']:.0%}"),
                  worst=_minutes(figures["worst_wait"]),
                  median=_minutes(figures["median_wait"]))
    cards = "".join(
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{value.format(**filled)}</div>'
        f'<div class="note">{note.format(**filled)}</div></div>'
        for label, value, note in CARDS)
    halts = "".join(f"<tr><td>{html.escape(name)}</td><td class='n'>{count}</td></tr>"
                    for name, count in figures["halts"].items()) or \
        "<tr><td>no halts in this window</td><td class='n'>0</td></tr>"
    decisions = "".join(
        f"<tr><td>{label}</td><td class='n'>{count}</td></tr>" for label, count in
        [("requested by the agent", figures["requested"]),
         ("approved by a person", figures["approved"]),
         ("refused by a person", figures["refused"]),
         ("still waiting", figures["waiting"])])
    warnings = ("".join(f"<li>{html.escape(line)}</li>" for line in said)
                if said else '<li class="calm">Every number is inside its '
                             'threshold. That is a statement about this window, '
                             'not about the system.</li>')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(figures['releases'][0] if figures['releases'] else 'agent')}</title>
<style>{STYLE}</style></head><body>
<h1>Flight agent — what happened</h1>
<p class="sub">{figures['runs']} runs &middot;
{html.escape(', '.join(figures['releases']))} &middot;
reply promised within {watch.reply_minutes:.0f} minutes</p>
<div class="cards">{cards}</div>
<h2>What needs a person</h2>
<ul class="alerts">{warnings}</ul>
<div class="row">
  <section><h2>Halts, by condition</h2>
    <table><tr><th>Condition</th><th>Runs</th></tr>{halts}</table></section>
  <section><h2>Actions a person decided</h2>
    <table><tr><th>Requests to move money</th><th>Count</th></tr>{decisions}</table>
  </section>
</div>
<footer>Read from <code>events.jsonl</code>. Every run above can be reopened by its
id: the record a person reads is on the run&rsquo;s <code>answered</code> line.</footer>
</body></html>
"""


def render(events: list, path: str, watch: Watch = Watch()) -> dict:
    figures = summarise(events)
    with open(path, "w", encoding="utf-8") as out:
        out.write(page(figures, alerts(figures, watch), watch))
    return figures


def show(figures: dict, said: list) -> None:
    print(f"  runs                {figures['runs']}  "
          f"({figures['delivered']} delivered, {figures['halted']} halted, "
          f"{figures['failed']} failed)")
    for name, count in figures["halts"].items():
        print(f"    halted on         {name}: {count}")
    print("  approval rate       "
          + ("—" if figures["approval_rate"] is None
             else f"{figures['approval_rate']:.0%}  "
                  f"({figures['approved']} of {figures['decided']} decided, "
                  f"{figures['waiting']} waiting)"))
    print(f"  escalation to reply median {_minutes(figures['median_wait'])}, "
          f"worst {_minutes(figures['worst_wait'])}, {figures['open']} open")
    print(f"  model calls per run {figures['calls_mean']:.1f} of {figures['ceiling']} "
          f"(worst run {figures['calls_max']})")
    print(f"  ran out of time     {figures['out_of_time']}")
    print("  what needs a person:")
    for line in said:
        print(f"    - {line}")


if __name__ == "__main__":
    from deploy import FOLDER, EventLog, Release

    events = EventLog(f"{FOLDER}/events.jsonl", Release()).read()
    figures = render(events, f"{FOLDER}/dashboard.html")
    print(f"=== {FOLDER}/events.jsonl, {len(events)} events ===")
    show(figures, alerts(figures))
    print(f"\n  wrote {FOLDER}/dashboard.html")
