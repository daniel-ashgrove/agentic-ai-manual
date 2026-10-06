"""
The same service around real agents.

Service, EventLog and dashboard.py do not change at all. An engine is anything
that takes a request and gives back Chapter 12's (run, composed) pair, so
swapping the scripted stand-ins for LangChain agents is this one function.

One thing does not survive the swap, and it is worth saying rather than
hiding: the per-delegation deadline. CombinedAgent builds its own runner
inside answer(), so there is no place to wrap it from out here, and no file
from an earlier chapter is edited to make room. What this engine can enforce
is the deadline a queue eats — a request that was already late when it reached
the front — which is the half of the problem that a busy service actually
hits. The other half is one parameter on CombinedAgent, and it is Part 2 of
this chapter's exercise.
"""

import os

from combined import Composition, Stores                                # Chapter 12
from combined_agent import CombinedAgent                                # Chapter 12
from deploy import Deadline, Release, Request                           # this chapter
from guardrails import Ledger, Limits                                   # Chapter 11
from tool_contracts import ToolRejected                                 # Chapter 5


def agent_engine(coordinator_model, worker_models: dict, critic_model,
                 max_steps: int = 25):
    """Build the models once; build one CombinedAgent per request, because a
    request's stores, budget and credit contract belong to that request."""
    def engine(request: Request, stores: Stores, ledger: Ledger, limits: Limits,
               composition: Composition, deadline: Deadline) -> tuple:
        if deadline.passed():
            deadline.refused("coordinator")
            raise ToolRejected("this request waited longer than its deadline before "
                               "it started; it was not run")
        system = CombinedAgent(coordinator_model, worker_models, critic_model,
                               ledger, stores, limits, composition, max_steps)
        return system.answer(request.question, request.traveller, request.day)
    return engine


LIVE = Release(version="1.0.0", model=os.environ.get("BOOK_MODEL", ""),
               provider="anthropic", prompt_revision="2026-06-09")


if __name__ == "__main__":
    import sys
    import threading
    from datetime import date

    from langchain_anthropic import ChatAnthropic

    from deploy import EventLog, Service, ask, counting_ids, serve
    from orchestration import WORKERS
    from policy_retrieval import QUESTION

    if not LIVE.model:
        sys.exit("BOOK_MODEL is not set. Set it to a current Claude model ID "
                 "and run this again.")
    model = ChatAnthropic(model=LIVE.model)
    log = EventLog("live/events.jsonl", LIVE)
    service = Service("live", log, Ledger(),
                      agent_engine(model, {name: model
                                           for name in list(WORKERS) + ["service"]},
                                   model), ids=counting_ids())
    server = serve(service, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(ask(port, "T-41", date.today().isoformat(), QUESTION))
    for written in log.read():
        print(written["run"], written["event"])
    server.shutdown()
